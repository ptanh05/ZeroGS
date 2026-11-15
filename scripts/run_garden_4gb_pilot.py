"""
ZeroGS: Garden 4GB Pilot Experiment Runner (Milestones 1,000 & 3,000 Steps).

Executes the pilot verification of ZeroGS-INT8 under a 4,096 MB VRAM hard budget constraint
on the unbounded outdoor Mip-NeRF 360 'garden' scene.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from argparse import ArgumentParser


def run_command(cmd, log_path=None):
    print(f"\n[RUN] {' '.join(cmd)}")
    start_time = time.time()
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    else:
        proc = subprocess.run(cmd, text=True)
    elapsed = time.time() - start_time
    if proc.returncode != 0:
        print(f"[ERROR] Command failed with code {proc.returncode} in {elapsed:.1f}s")
    else:
        print(f"[SUCCESS] Command completed in {elapsed:.1f}s")
    return proc.returncode, elapsed


def parse_telemetry(telemetry_file):
    """Extract peak allocated, peak reserved, and admission telemetry."""
    peak_allocated_mb = 0.0
    peak_reserved_mb = 0.0
    final_gaussians = 0
    total_evaluations = 0
    prune_first_triggers = 0
    shrink_triggers = 0
    pruned_via_admission = 0

    if not os.path.exists(telemetry_file):
        return {}

    with open(telemetry_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                event = rec.get("event")
                if event == "admission_eval":
                    total_evaluations += 1
                    cur_mb = rec.get("current_allocated_mb", 0.0)
                    peak_allocated_mb = max(peak_allocated_mb, cur_mb)
                    action = rec.get("action")
                    if action == "PRUNE_FIRST":
                        prune_first_triggers += 1
                    elif action == "SHRINK_AND_PRUNE_FIRST":
                        shrink_triggers += 1
                    pruned_via_admission += rec.get("pruned_count", 0)
                elif "memory" in rec:
                    mem = rec["memory"]
                    alloc_mb = float(mem.get("peak_allocated_mb", mem.get("allocated_bytes", 0) / (1024**2)))
                    res_mb = float(mem.get("peak_reserved_mb", mem.get("reserved_bytes", 0) / (1024**2)))
                    peak_allocated_mb = max(peak_allocated_mb, alloc_mb)
                    peak_reserved_mb = max(peak_reserved_mb, res_mb)
                if "admission" in rec:
                    adm = rec["admission"]
                    total_evaluations = max(total_evaluations, adm.get("total_evaluations", 0))
                    prune_first_triggers = max(prune_first_triggers, adm.get("prune_first_triggered", 0))
                    shrink_triggers = max(shrink_triggers, adm.get("shrink_triggered", 0))
                    pruned_via_admission = max(pruned_via_admission, adm.get("total_pruned_via_admission", 0))
                if "final_gaussians" in rec:
                    final_gaussians = rec["final_gaussians"]
                elif "num_gaussians" in rec:
                    final_gaussians = rec["num_gaussians"]
            except Exception:
                continue

    return {
        "peak_allocated_mb": round(peak_allocated_mb, 2),
        "peak_reserved_mb": round(peak_reserved_mb, 2),
        "final_gaussians": final_gaussians,
        "admission_evaluations": total_evaluations,
        "prune_first_triggers": prune_first_triggers,
        "shrink_triggers": shrink_triggers,
        "pruned_via_admission": pruned_via_admission,
    }


def verify_checkpoint_files(model_path, iteration):
    import numpy as np
    from plyfile import PlyData

    pc_dir = Path(model_path) / "point_cloud" / f"iteration_{iteration}"
    ply_path = pc_dir / "point_cloud.ply"
    splat_path = pc_dir / "point_cloud.splat"
    npz_path = pc_dir / "point_cloud_compressed.npz"

    res = {
        "ply_exists": ply_path.exists(),
        "ply_size_mb": round(ply_path.stat().st_size / (1024**2), 2) if ply_path.exists() else 0.0,
        "ply_valid": False,
        "ply_points": 0,
        "has_nan_inf": False,
        "splat_exists": splat_path.exists(),
        "splat_size_mb": round(splat_path.stat().st_size / (1024**2), 2) if splat_path.exists() else 0.0,
        "splat_valid": False,
        "splat_points": 0,
        "npz_exists": npz_path.exists(),
        "npz_size_mb": round(npz_path.stat().st_size / (1024**2), 2) if npz_path.exists() else 0.0,
        "npz_valid": False,
        "npz_points": 0,
    }

    if ply_path.exists():
        try:
            plydata = PlyData.read(str(ply_path))
            v = plydata["vertex"]
            res["ply_points"] = len(v)
            res["ply_valid"] = res["ply_points"] > 0
            # NaN/Inf check on positions
            x = np.asarray(v["x"])
            y = np.asarray(v["y"])
            z = np.asarray(v["z"])
            if np.isnan(x).any() or np.isnan(y).any() or np.isnan(z).any() or \
               np.isinf(x).any() or np.isinf(y).any() or np.isinf(z).any():
                res["has_nan_inf"] = True
        except Exception as e:
            res["ply_error"] = str(e)

    if splat_path.exists():
        try:
            size = splat_path.stat().st_size
            res["splat_points"] = size // 32
            res["splat_valid"] = (size % 32 == 0) and (res["splat_points"] == res["ply_points"])
        except Exception as e:
            res["splat_error"] = str(e)

    if npz_path.exists():
        try:
            data = np.load(str(npz_path))
            res["npz_points"] = int(data["xyz"].shape[0])
            res["npz_valid"] = res["npz_points"] == res["ply_points"]
        except Exception as e:
            res["npz_error"] = str(e)

    if res["ply_size_mb"] > 0:
        res["splat_compression_ratio"] = round(res["ply_size_mb"] / max(res["splat_size_mb"], 0.001), 2)
        res["npz_compression_ratio"] = round(res["ply_size_mb"] / max(res["npz_size_mb"], 0.001), 2)

    return res


def main():
    parser = ArgumentParser(description="ZeroGS Garden 4GB Pilot Experiment Runner")
    parser.add_argument("--source_path", type=str, default="data/garden", help="Scene path")
    parser.add_argument("--images", type=str, default="images_4", help="Images directory")
    parser.add_argument("--iterations", type=int, default=3000, help="Training iterations")
    parser.add_argument("--output_dir", type=str, default="experiments/results/garden_4gb/garden_zerogs_int8_pilot")
    args = parser.parse_args()

    python_exe = sys.executable
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    telemetry_file = out_dir / "telemetry.jsonl"
    train_log = out_dir / "train.log"
    render_log = out_dir / "render.log"
    metrics_file = out_dir / "metrics_3000.json"
    summary_file = out_dir / "pilot_summary.json"

    print("==================================================================")
    print(" ZERO-GS: GARDEN 4GB PILOT EXPERIMENT")
    print(f" Scene: {args.source_path} ({args.images})")
    print(f" Target Budget: 4096 MB Hard Cap (PyTorch Budget: 3584 MB)")
    print(f" Optimizer: Int8GaussianAdam (Logarithmic uint8 dynamic range)")
    print(f" Iterations: {args.iterations} (Checkpoints at 1000 and 3000)")
    print(f" Output Directory: {out_dir}")
    print("==================================================================")

    # 1. Training Phase
    train_cmd = [
        python_exe, "train.py",
        "-s", args.source_path,
        "-i", args.images,
        "-m", str(out_dir),
        "--eval",
        "--iterations", str(args.iterations),
        "--save_iterations", "1000", str(args.iterations),
        "--test_iterations", "1000", str(args.iterations),
        "--checkpoint_iterations", "1000", str(args.iterations),
        "--seed", "0",
        "--hard_vram_limit_mb", "4096.0",
        "--safety_headroom_mb", "256.0",
        "--non_pytorch_allowance_mb", "256.0",
        "--optimizer_type", "int8",
        "--pruning_mode", "legacy_as_is",
        "--allocator_strategy", "gain_per_byte",
        "--inflow_mode", "dynamic",
        "--inflow_cap_min", "5000",
        "--inflow_cap_max", "15000",
        "--target_headroom_mb", "1024.0",
        "--kappa_split", "0.40",
        "--tau_age", "500",
        "--tau_reset", "400",
        "--data_device", "cpu",
        "--cost_calibration", "experiments/results/cost_calibration.json",
        "--telemetry_jsonl", str(telemetry_file),
    ]

    train_code, train_elapsed = run_command(train_cmd, log_path=str(train_log))

    # 2. Rendering Holdout Views at Iteration 3000
    render_elapsed = 0.0
    if train_code == 0:
        render_cmd = [
            python_exe, "render.py",
            "-s", args.source_path,
            "-m", str(out_dir),
            "-i", args.images,
            "--iteration", str(args.iterations),
            "--eval",
            "--data_device", "cpu",
            "--skip_train",
        ]
        render_code, render_elapsed = run_command(render_cmd, log_path=str(render_log))

        # 3. Metrics Evaluation
        if render_code == 0:
            metrics_cmd = [
                python_exe, "metrics.py",
                "-m", str(out_dir),
                "-o", str(metrics_file),
            ]
            run_command(metrics_cmd)

    # 4. Telemetry and File Validation
    telemetry = parse_telemetry(str(telemetry_file))
    chk_1000 = verify_checkpoint_files(str(out_dir), 1000)
    chk_3000 = verify_checkpoint_files(str(out_dir), 3000)

    # Parse Metrics
    metrics_3000 = {}
    if metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                raw_m = json.load(f)
                if isinstance(raw_m, dict):
                    metrics_3000 = list(raw_m.values())[0] if len(raw_m) == 1 else raw_m
        except Exception as e:
            metrics_3000 = {"error": str(e)}

    # Export compact splat and npz for both checkpoints if missing
    for it in [1000, 3000]:
        pc_dir = out_dir / "point_cloud" / f"iteration_{it}"
        ply_f = pc_dir / "point_cloud.ply"
        splat_f = pc_dir / "point_cloud.splat"
        npz_f = pc_dir / "point_cloud_compressed.npz"
        if ply_f.exists() and not splat_f.exists():
            try:
                from utils.general_utils import export_to_splat, export_to_npz
                export_to_splat(str(ply_f), str(splat_f))
                export_to_npz(str(ply_f), str(npz_f))
                print(f"[EXPORT] Generated .splat and .npz for iteration {it}")
            except Exception as e:
                print(f"[EXPORT WARNING] Failed export for iteration {it}: {e}")

    summary = {
        "scene": args.source_path,
        "images": args.images,
        "iterations": args.iterations,
        "train_code": train_code,
        "train_elapsed_sec": round(train_elapsed, 2),
        "render_elapsed_sec": round(render_elapsed, 2),
        "peak_allocated_mb": telemetry.get("peak_allocated_mb", 0.0),
        "peak_reserved_mb": telemetry.get("peak_reserved_mb", 0.0),
        "hard_vram_limit_mb": 4096.0,
        "budget_preserved": telemetry.get("peak_reserved_mb", 0.0) <= 4096.0,
        "admission_evaluations": telemetry.get("admission_evaluations", 0),
        "prune_first_triggers": telemetry.get("prune_first_triggers", 0),
        "shrink_triggers": telemetry.get("shrink_triggers", 0),
        "pruned_via_admission": telemetry.get("pruned_via_admission", 0),
        "checkpoint_1000": chk_1000,
        "checkpoint_3000": chk_3000,
        "metrics_3000": metrics_3000,
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n==================================================================")
    print(" PILOT RUN COMPLETED")
    print(f" Train Exit Code: {train_code}")
    print(f" Peak Reserved VRAM: {summary['peak_reserved_mb']} MB (Cap: 4096.0 MB, Preserved: {summary['budget_preserved']})")
    print(f" Gaussians at 1000: {chk_1000.get('ply_points', 0)} | at 3000: {chk_3000.get('ply_points', 0)}")
    print(f" Admission Evals: {summary['admission_evaluations']} | Prune-First Triggers: {summary['prune_first_triggers']}")
    if metrics_3000:
        print(f" 3000-Step Metrics -> PSNR: {metrics_3000.get('PSNR', 'N/A')} | SSIM: {metrics_3000.get('SSIM', 'N/A')} | LPIPS: {metrics_3000.get('LPIPS', 'N/A')}")
    print(f" Summary written to: {summary_file}")
    print("==================================================================")


if __name__ == "__main__":
    main()

