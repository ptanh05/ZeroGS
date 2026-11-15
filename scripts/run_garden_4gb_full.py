"""
ZeroGS: Garden 4GB Full Convergence Runner (Resuming from Step 3,000 to 7,000 and 30,000).

Strictly enforces:
- Hard VRAM Budget: 4096.0 MB (PyTorch budget: 3584 MB)
- Continuation from: experiments/results/garden_4gb/garden_zerogs_int8_pilot/chkpnt3000.pth
- Output Directory: experiments/results/garden_4gb/garden_zerogs_int8_full
- Optimizer: Int8GaussianAdam (Logarithmic uint8 dynamic range)
- Evaluates at: 7k, 15k, 30k across 24 test holdout views.
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
        with open(log_path, "a", encoding="utf-8") as f:
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
    """Extract telemetry metrics from jsonl log."""
    peak_allocated_mb = 0.0
    peak_reserved_mb = 0.0
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
            except Exception:
                continue

    return {
        "peak_allocated_mb": round(peak_allocated_mb, 2),
        "peak_reserved_mb": round(peak_reserved_mb, 2),
        "admission_evaluations": total_evaluations,
        "prune_first_triggers": prune_first_triggers,
        "shrink_triggers": shrink_triggers,
        "pruned_via_admission": pruned_via_admission,
    }


def verify_checkpoint(model_path, iteration):
    from plyfile import PlyData
    import numpy as np

    pc_dir = Path(model_path) / "point_cloud" / f"iteration_{iteration}"
    ply_path = pc_dir / "point_cloud.ply"
    res = {"exists": ply_path.exists(), "points": 0, "size_mb": 0.0, "has_nan_inf": False}
    if ply_path.exists():
        res["size_mb"] = round(ply_path.stat().st_size / (1024**2), 2)
        try:
            pd = PlyData.read(str(ply_path))
            v = pd["vertex"]
            res["points"] = len(v)
            x, y, z = np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])
            if np.isnan(x).any() or np.isnan(y).any() or np.isnan(z).any() or \
               np.isinf(x).any() or np.isinf(y).any() or np.isinf(z).any():
                res["has_nan_inf"] = True
        except Exception:
            pass
    return res


def evaluate_milestone(python_exe, scene_path, images, out_dir, iteration):
    """Renders holdout test views and computes PSNR, SSIM, LPIPS."""
    render_cmd = [
        python_exe, "render.py",
        "-s", scene_path,
        "-m", str(out_dir),
        "-i", images,
        "--iteration", str(iteration),
        "--eval",
        "--data_device", "cpu",
        "--skip_train",
    ]
    rcode, rtime = run_command(render_cmd)
    metrics_path = out_dir / f"metrics_{iteration}.json"
    if rcode == 0:
        metrics_cmd = [
            python_exe, "metrics.py",
            "-m", str(out_dir),
            "-o", str(metrics_path),
        ]
        run_command(metrics_cmd)

    metrics = {}
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                raw_m = json.load(f)
                metrics = list(raw_m.values())[0] if len(raw_m) == 1 else raw_m
        except Exception:
            pass
    return metrics


def export_compact_formats(model_path, iteration):
    pc_dir = Path(model_path) / "point_cloud" / f"iteration_{iteration}"
    ply_f = pc_dir / "point_cloud.ply"
    splat_f = pc_dir / "point_cloud.splat"
    npz_f = pc_dir / "point_cloud_compressed.npz"
    if ply_f.exists():
        try:
            from utils.general_utils import export_to_splat, export_to_npz
            export_to_splat(str(ply_f), str(splat_f))
            export_to_npz(str(ply_f), str(npz_f))
            print(f"[EXPORT] Generated .splat and .npz for iteration {iteration}")
        except Exception as e:
            print(f"[EXPORT ERROR] {e}")


def main():
    parser = ArgumentParser(description="ZeroGS Garden 4GB Full Convergence Runner")
    parser.add_argument("--source_path", type=str, default="data/garden")
    parser.add_argument("--images", type=str, default="images_4")
    parser.add_argument("--checkpoint", type=str, default="experiments/results/garden_4gb/garden_zerogs_int8_pilot/chkpnt3000.pth")
    parser.add_argument("--output_dir", type=str, default="experiments/results/garden_4gb/garden_zerogs_int8_full")
    parser.add_argument("--max_iterations", type=int, default=30000)
    args = parser.parse_args()

    python_exe = sys.executable
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    telemetry_file = out_dir / "telemetry.jsonl"
    train_log = out_dir / "train.log"

    print("==================================================================")
    print(" ZERO-GS: GARDEN 4GB FULL CONVERGENCE TRAINING")
    print(f" Resuming from: {args.checkpoint}")
    print(f" Target Iterations: {args.max_iterations} (Milestones at 7k, 15k, 30k)")
    print(f" Output Directory: {out_dir}")
    print(f" Budget Cap: 4096.0 MB (PyTorch: 3584 MB)")
    print("==================================================================")

    # Base training command
    cmd = [
        python_exe, "train.py",
        "-s", args.source_path,
        "-i", args.images,
        "-m", str(out_dir),
        "--eval",
        "--start_checkpoint", str(args.checkpoint),
        "--iterations", str(args.max_iterations),
        "--save_iterations", "7000", "15000", str(args.max_iterations),
        "--test_iterations", "7000", "15000", str(args.max_iterations),
        "--checkpoint_iterations", "7000", "15000", str(args.max_iterations),
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

    code, elapsed = run_command(cmd, log_path=str(train_log))
    print(f"[TRAIN FINISHED] Code: {code}, Time: {elapsed:.1f}s")

    # Evaluate milestones that were generated
    results = {}
    for it in [7000, 15000, args.max_iterations]:
        chk = verify_checkpoint(str(out_dir), it)
        if chk["exists"]:
            print(f"\n[EVALUATION] Evaluating Milestone {it}...")
            metrics = evaluate_milestone(python_exe, args.source_path, args.images, out_dir, it)
            export_compact_formats(str(out_dir), it)
            results[str(it)] = {
                "checkpoint": chk,
                "metrics": metrics,
            }

    # Summary
    telem = parse_telemetry(str(telemetry_file))
    summary = {
        "scene": args.source_path,
        "resumed_from": args.checkpoint,
        "exit_code": code,
        "elapsed_sec": round(elapsed, 2),
        "telemetry": telem,
        "milestones": results,
    }
    with open(out_dir / "convergence_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # CSV Generation
    csv_path = out_dir / "GARDEN_4GB_CONVERGENCE.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Iteration,PSNR,SSIM,LPIPS,Primitives,Peak_Allocated_MB,Peak_Reserved_MB,Budget_Preserved\n")
        f.write("3000,14.5879,0.2992,0.7402,191133,262.28,2376.0,True\n")
        for it, data in results.items():
            m = data.get("metrics", {})
            c = data.get("checkpoint", {})
            psnr = m.get("PSNR", "N/A")
            ssim = m.get("SSIM", "N/A")
            lpips = m.get("LPIPS", "N/A")
            pts = c.get("points", 0)
            alloc = telem.get("peak_allocated_mb", 0.0)
            res = telem.get("peak_reserved_mb", 0.0)
            pres = res <= 4096.0
            f.write(f"{it},{psnr},{ssim},{lpips},{pts},{alloc},{res},{pres}\n")

    print(f"\n[DONE] Convergence results saved to: {out_dir / 'convergence_summary.json'} and {csv_path}")


if __name__ == "__main__":
    main()

