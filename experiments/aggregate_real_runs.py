"""Aggregate measured ZeroGS runs without inventing missing metrics.

Each run directory must contain run_manifest.json and run_status.json. Completed
runs should also contain render-generated results.json files and
training_telemetry.jsonl. Simulation manifests are rejected by default.
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def find_metric_result(run_dir: Path) -> Dict[str, Any]:
    candidates = sorted(run_dir.rglob("results.json"))
    test_candidates = [path for path in candidates if "test" in path.parts]
    selected = test_candidates[-1] if test_candidates else (candidates[-1] if candidates else None)
    return load_json(selected) if selected else {}


def telemetry_peak_reserved_mb(path: Path) -> float:
    peak = 0.0
    if not path.exists():
        return peak
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            for key in ("memory_before", "memory_after", "memory"):
                memory = record.get(key) or {}
                peak = max(peak, float(memory.get("peak_reserved_mb", 0.0)))
    return peak


def collect_runs(root: Path, allow_simulation: bool = False) -> List[Dict[str, Any]]:
    records = []
    for manifest_path in root.rglob("run_manifest.json"):
        run_dir = manifest_path.parent
        manifest = load_json(manifest_path)
        if manifest.get("evidence_type") != "measured" and not allow_simulation:
            continue
        status_path = run_dir / "run_status.json"
        status = load_json(status_path) if status_path.exists() else {"status": "unknown", "oom": False}
        metrics = find_metric_result(run_dir) if status.get("status") == "completed" else {}
        records.append({
            "run_dir": str(run_dir),
            "run_id": manifest.get("run_id"),
            "evidence_type": manifest.get("evidence_type", "unknown"),
            "method": manifest.get("method", "unknown"),
            "scene": manifest.get("scene", "unknown"),
            "budget_mb": float(manifest.get("hard_vram_limit_mb", 0.0)),
            "seed": int(manifest.get("seed", 0)),
            "status": status.get("status", "unknown"),
            "oom": bool(status.get("oom", False)),
            "budget_exceeded": bool(status.get("budget_exceeded", False)),
            "psnr": metrics.get("PSNR"),
            "ssim": metrics.get("SSIM"),
            "lpips": metrics.get("LPIPS"),
            "peak_reserved_mb": telemetry_peak_reserved_mb(run_dir / "training_telemetry.jsonl"),
        })
    return records


def numeric_summary(values: Iterable[Any]) -> Dict[str, Any]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": mean(clean),
        "std": stdev(clean) if len(clean) > 1 else 0.0,
        "n": len(clean),
    }


def aggregate(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for record in records:
        groups[(record["scene"], record["method"], record["budget_mb"])].append(record)

    summaries = []
    for (scene, method, budget_mb), group in sorted(groups.items()):
        completed = [row for row in group if row["status"] == "completed"]
        summaries.append({
            "scene": scene,
            "method": method,
            "budget_mb": budget_mb,
            "runs": len(group),
            "completed_runs": len(completed),
            "oom_runs": sum(bool(row["oom"]) for row in group),
            "budget_exceeded_runs": sum(bool(row["budget_exceeded"]) for row in group),
            "psnr": numeric_summary(row["psnr"] for row in completed),
            "ssim": numeric_summary(row["ssim"] for row in completed),
            "lpips": numeric_summary(row["lpips"] for row in completed),
            "peak_reserved_mb": numeric_summary(row["peak_reserved_mb"] for row in completed),
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate measured ZeroGS experiment runs")
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("experiments/results/real"))
    parser.add_argument("--allow_simulation", action="store_true")
    args = parser.parse_args()

    records = collect_runs(args.runs_root, allow_simulation=args.allow_simulation)
    if not records:
        raise SystemExit("No eligible run manifests found; measured evidence was not fabricated.")
    summaries = aggregate(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.output_dir / "runs.json").open("w", encoding="utf-8") as output:
        json.dump(records, output, indent=2)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as output:
        json.dump(summaries, output, indent=2)

    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as output:
        fieldnames = [
            "scene", "method", "budget_mb", "runs", "completed_runs", "oom_runs", "budget_exceeded_runs",
            "psnr_mean", "psnr_std", "ssim_mean", "ssim_std",
            "lpips_mean", "lpips_std", "peak_reserved_mb_mean", "peak_reserved_mb_std",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({
                "scene": row["scene"], "method": row["method"], "budget_mb": row["budget_mb"],
                "runs": row["runs"], "completed_runs": row["completed_runs"], "oom_runs": row["oom_runs"],
                "budget_exceeded_runs": row["budget_exceeded_runs"],
                "psnr_mean": row["psnr"]["mean"], "psnr_std": row["psnr"]["std"],
                "ssim_mean": row["ssim"]["mean"], "ssim_std": row["ssim"]["std"],
                "lpips_mean": row["lpips"]["mean"], "lpips_std": row["lpips"]["std"],
                "peak_reserved_mb_mean": row["peak_reserved_mb"]["mean"],
                "peak_reserved_mb_std": row["peak_reserved_mb"]["std"],
            })

    print(f"Aggregated {len(records)} measured runs into {args.output_dir}")


if __name__ == "__main__":
    main()
