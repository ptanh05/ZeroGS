"""Launch measured P3 allocator and S1 occlusion ablations on real scenes."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from run_real_matrix import parse_scene, run_command


ABLATIONS = {
    "p3_gain_per_byte": ["--allocator_strategy", "gain_per_byte", "--occlusion_mode", "full"],
    "p3_demand_only": ["--allocator_strategy", "demand_only", "--occlusion_mode", "full"],
    "p3_random": ["--allocator_strategy", "random", "--occlusion_mode", "full"],
    "s1_visibility_only": [
        "--allocator_strategy", "gain_per_byte", "--occlusion_mode", "none",
        "--disable_structure_protection",
    ],
    "s1_transmittance_only": [
        "--allocator_strategy", "gain_per_byte", "--occlusion_mode", "transmittance_only",
        "--disable_structure_protection",
    ],
    "s1_depth_only": [
        "--allocator_strategy", "gain_per_byte", "--occlusion_mode", "depth_only",
        "--disable_structure_protection",
    ],
    "s1_full_no_protection": [
        "--allocator_strategy", "gain_per_byte", "--occlusion_mode", "full",
        "--disable_structure_protection",
    ],
    "s1_full": ["--allocator_strategy", "gain_per_byte", "--occlusion_mode", "full"],
}


def build_ablation_commands(
    repo_root: Path,
    scene_name: str,
    scene_path: Path,
    model_path: Path,
    budget_mb: float,
    seed: int,
    iterations: int,
    calibration: Path,
    ablation_name: str,
    images_dir: str = "images_4",
    densify_from_iter: int = 500,
    densify_until_iter: int = 15_000,
    densification_interval: int = 100,
    densify_grad_threshold: float = 0.0002,
) -> List[List[str]]:
    if ablation_name not in ABLATIONS:
        raise ValueError(f"Unknown ablation: {ablation_name}")
    python = sys.executable
    train = [
        python, str(repo_root / "train.py"),
        "-s", str(scene_path), "-m", str(model_path), "-i", images_dir, "--eval",
        "--iterations", str(iterations),
        "--save_iterations", str(iterations),
        "--test_iterations", str(iterations),
        "--hard_vram_limit_mb", str(budget_mb),
        "--seed", str(seed),
        "--cost_calibration", str(calibration),
        "--experiment_method", ablation_name,
        "--experiment_scene", scene_name,
        "--densify_from_iter", str(densify_from_iter),
        "--densify_until_iter", str(densify_until_iter),
        "--densification_interval", str(densification_interval),
        "--densify_grad_threshold", str(densify_grad_threshold),
        *ABLATIONS[ablation_name],
    ]
    render = [
        python, str(repo_root / "render.py"),
        "-s", str(scene_path), "-m", str(model_path), "-i", images_dir, "--eval",
        "--iteration", str(iterations), "--skip_train",
    ]
    metrics = [python, str(repo_root / "metrics.py"), "-m", str(model_path)]
    return [train, render, metrics]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run measured ZeroGS P3/S1 ablations")
    parser.add_argument("--scene", action="append", type=parse_scene, required=True, help="NAME=PATH")
    parser.add_argument("--ablations", nargs="+", choices=sorted(ABLATIONS), default=list(ABLATIONS))
    parser.add_argument("--budget", type=float, default=6144.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=Path("output/real_ablation"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--images", default="images_4", help="Scene image directory")
    parser.add_argument("--densify_from_iter", type=int, default=500)
    parser.add_argument("--densify_until_iter", type=int, default=15_000)
    parser.add_argument("--densification_interval", type=int, default=100)
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0002)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    calibration = args.calibration.expanduser().resolve()
    if not calibration.exists():
        raise SystemExit(f"Calibration does not exist: {calibration}")
    plan = []
    for scene_name, scene_path in args.scene:
        for ablation in args.ablations:
            for seed in args.seeds:
                run_dir = args.output_root / scene_name / f"{ablation}_seed{seed}"
                plan.append({
                    "scene": scene_name,
                    "ablation": ablation,
                    "seed": seed,
                    "budget_mb": args.budget,
                    "run_dir": str(run_dir),
                    "commands": build_ablation_commands(
                        repo_root, scene_name, scene_path, run_dir, args.budget,
                        seed, args.iterations, calibration, ablation,
                        args.images,
                        args.densify_from_iter,
                        args.densify_until_iter,
                        args.densification_interval,
                        args.densify_grad_threshold,
                    ),
                })

    args.output_root.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_root / "ablation_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Prepared {len(plan)} measured ablation runs in {plan_path}")
    if not args.execute:
        print("Dry run only. Re-run with --execute to launch sequential GPU jobs.")
        return

    for item in plan:
        run_dir = Path(item["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        for stage, command in zip(("train", "render", "metrics"), item["commands"]):
            print(f"[{item['scene']} {item['ablation']} seed={item['seed']}] {stage}")
            if run_command(command, run_dir / f"{stage}.log") != 0:
                print(f"Stage {stage} failed; continuing to the next run.")
                break


if __name__ == "__main__":
    main()
