"""
ZeroGS 4GB Empirical Matrix Runner
Automates training, evaluation, metrics extraction, and format validation
across Vanilla 3DGS, ZeroGS-FP16, and ZeroGS-INT8 under a 4096 MB VRAM budget constraint.
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
    """Extract peak allocated and peak reserved VRAM from jsonl telemetry."""
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
                    if "peak_allocated_mb" in mem:
                        alloc_mb = float(mem["peak_allocated_mb"])
                    else:
                        alloc_mb = mem.get("allocated_bytes", 0) / (1024**2)
                    if "peak_reserved_mb" in mem:
                        res_mb = float(mem["peak_reserved_mb"])
                    else:
                        res_mb = mem.get("reserved_bytes", 0) / (1024**2)
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
            res["ply_points"] = len(plydata["vertex"])
            res["ply_valid"] = res["ply_points"] > 0
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
    parser = ArgumentParser(description="Run 4GB Empirical Benchmark Matrix")
    parser.add_argument("--source_path", "-s", default="data/bonsai")
    parser.add_argument("--images", "-i", default="images_4")
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--output_dir", default="experiments/results/matrix_4gb")
    parser.add_argument("--skip_vanilla", action="store_true", default=False)
    parser.add_argument("--skip_fp32", action="store_true", default=False)
    parser.add_argument("--skip_fp16", action="store_true", default=False)
    parser.add_argument("--skip_int8", action="store_true", default=False)
    args = parser.parse_args()

    python_exe = sys.executable
    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    scene_name = Path(args.source_path).name

    matrix = []
    if not args.skip_int8:
        matrix.append({
            "name": "ZeroGS_INT8",
            "model_path": str(out_base / f"{scene_name}_zerogs_int8"),
            "optimizer_type": "int8_adam",
            "flags": [
                "--hard_vram_limit_mb", "4096.0",
                "--safety_headroom_mb", "256.0",
                "--non_pytorch_allowance_mb", "256.0",
                "--pruning_mode", "legacy_as_is",
            ],
            "description": "ZeroGS with Int8GaussianAdam and Preservation Lifecycle under 4096MB hard budget",
        })
    if not args.skip_vanilla:
        matrix.append({
            "name": "Vanilla_FP32",
            "model_path": str(out_base / f"{scene_name}_vanilla_fp32"),
            "optimizer_type": "default",
            "flags": ["--disable_zero_oom"],
            "description": "Vanilla 3DGS baseline with default FP32 Adam (unconstrained)",
        })
    if not args.skip_fp32:
        matrix.append({
            "name": "ZeroGS_FP32",
            "model_path": str(out_base / f"{scene_name}_zerogs_fp32"),
            "optimizer_type": "default",
            "flags": [
                "--hard_vram_limit_mb", "4096.0",
                "--safety_headroom_mb", "256.0",
                "--non_pytorch_allowance_mb", "256.0",
                "--pruning_mode", "legacy_as_is",
            ],
            "description": "ZeroGS with default FP32 Adam and Preservation Lifecycle under 4096MB hard budget",
        })
    if not args.skip_fp16:
        matrix.append({
            "name": "ZeroGS_FP16",
            "model_path": str(out_base / f"{scene_name}_zerogs_fp16"),
            "optimizer_type": "half_adam",
            "flags": [
                "--hard_vram_limit_mb", "4096.0",
                "--safety_headroom_mb", "256.0",
                "--non_pytorch_allowance_mb", "256.0",
                "--pruning_mode", "legacy_as_is",
            ],
            "description": "ZeroGS with HalfPrecisionAdam and Preservation Lifecycle under 4096MB hard budget",
        })

    summary_path = out_base / "matrix_summary.json"
    summary_results = {}
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_results = json.load(f)
        except Exception:
            summary_results = {}

    for exp in matrix:
        name = exp["name"]
        m_path = exp["model_path"]
        opt_type = exp["optimizer_type"]
        extra_flags = exp["flags"]

        print(f"\n=======================================================")
        print(f" STARTING EXPERIMENT: {name}")
        print(f" Optimizer: {opt_type} | Budget Cap: 4096 MB | Iterations: {args.iterations}")
        print(f" Destination: {m_path}")
        print(f"=======================================================")

        # 1. Train
        train_log = str(out_base / f"{name}_train.log")
        train_cmd = [
            python_exe, "train.py",
            "-s", args.source_path,
            "-m", m_path,
            "-i", args.images,
            "--iterations", str(args.iterations),
            "--eval",
            "--test_iterations", str(args.iterations),
            "--save_iterations", str(args.iterations),
            "--optimizer_type", opt_type,
        ] + extra_flags

        train_code, train_elapsed = run_command(train_cmd, log_path=train_log)

        # 2. Render Test Views
        render_code = -1
        render_elapsed = 0.0
        metrics_file = os.path.join(m_path, "metrics.json")
        if train_code == 0:
            render_log = str(out_base / f"{name}_render.log")
            render_cmd = [
                python_exe, "render.py",
                "-s", args.source_path,
                "-m", m_path,
                "-i", args.images,
                "--iteration", str(args.iterations),
                "--eval",
                "--skip_train",
            ]
            render_code, render_elapsed = run_command(render_cmd, log_path=render_log)

            # 3. Evaluate Metrics (PSNR, SSIM, LPIPS)
            if render_code == 0:
                metrics_cmd = [
                    python_exe, "metrics.py",
                    "-m", m_path,
                    "-o", metrics_file,
                ]
                metrics_code, _ = run_command(metrics_cmd)

        metrics = {}
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    raw_metrics = json.load(f)
                    if isinstance(raw_metrics, dict):
                        if len(raw_metrics) == 1:
                            metrics = list(raw_metrics.values())[0]
                        else:
                            metrics = raw_metrics
            except Exception as e:
                print(f"[WARN] Failed to load metrics: {e}")

        # 4. Telemetry and File Validation
        telemetry_file = os.path.join(m_path, "training_telemetry.jsonl")
        telemetry_stats = parse_telemetry(telemetry_file)

        # 5. Check run_status.json
        status_file = os.path.join(m_path, "run_status.json")
        status = {}
        if os.path.exists(status_file):
            try:
                with open(status_file, "r", encoding="utf-8") as f:
                    status = json.load(f)
            except Exception:
                pass

        file_validation = verify_checkpoint_files(m_path, args.iterations)

        summary_results[name] = {
            "name": name,
            "description": exp["description"],
            "status": status.get("status", "unknown"),
            "oom": status.get("oom", False),
            "training_time_sec": round(train_elapsed, 1),
            "training_speed_it_per_sec": round(args.iterations / max(train_elapsed, 0.1), 2),
            "metrics": metrics,
            "telemetry": telemetry_stats,
            "files": file_validation,
        }

        print(f"\n[DONE {name}] Time: {train_elapsed:.1f}s | Status: {status.get('status')} | Metrics: {metrics}")

    # Write overall summary JSON
    summary_path = out_base / "matrix_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    print(f"\n=======================================================")
    print(f" ALL MATRIX EXPERIMENTS COMPLETED!")
    print(f" Summary saved to: {summary_path}")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
