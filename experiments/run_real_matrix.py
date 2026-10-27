"""Run a reproducible ZeroGS scene × budget × seed matrix on real data."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_scene(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Scene must use NAME=PATH syntax")
    name, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not name or not path.exists():
        raise argparse.ArgumentTypeError(f"Invalid scene mapping: {value}")
    return name, path


def build_run_commands(
    repo_root: Path,
    scene_name: str,
    scene_path: Path,
    model_path: Path,
    budget_mb: float,
    seed: int,
    iterations: int,
    calibration_path: Path,
    images_dir: str = "images_4",
) -> List[List[str]]:
    python = sys.executable
    train = [
        python, str(repo_root / "train.py"),
        "-s", str(scene_path),
        "-m", str(model_path),
        "-i", images_dir,
        "--eval",
        "--iterations", str(iterations),
        "--save_iterations", str(iterations),
        "--test_iterations", str(iterations),
        "--hard_vram_limit_mb", str(budget_mb),
        "--seed", str(seed),
        "--cost_calibration", str(calibration_path),
        "--experiment_method", "ZeroGS",
        "--experiment_scene", scene_name,
    ]
    render = [
        python, str(repo_root / "render.py"),
        "-s", str(scene_path),
        "-m", str(model_path),
        "-i", images_dir,
        "--eval",
        "--iteration", str(iterations),
        "--skip_train",
    ]
    metrics = [python, str(repo_root / "metrics.py"), "-m", str(model_path)]
    return [train, render, metrics]


def run_command(command: List[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(process.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run measured ZeroGS benchmark matrix")
    parser.add_argument("--scene", action="append", type=parse_scene, required=True, help="NAME=PATH")
    parser.add_argument("--budgets", nargs="+", type=float, default=[4096.0, 6144.0, 8192.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=Path("output/real_benchmark"))
    parser.add_argument("--execute", action="store_true", help="Actually launch GPU jobs")
    parser.add_argument("--images", default="images_4", help="Scene image directory")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    calibration = args.calibration.expanduser().resolve()
    if not calibration.exists():
        raise SystemExit(f"Calibration does not exist: {calibration}")

    plan = []
    for scene_name, scene_path in args.scene:
        for budget in args.budgets:
            for seed in args.seeds:
                run_dir = args.output_root / scene_name / f"zerogs_{int(budget)}mb_seed{seed}"
                commands = build_run_commands(
                    repo_root, scene_name, scene_path, run_dir,
                    budget, seed, args.iterations, calibration,
                    args.images,
                )
                plan.append({
                    "scene": scene_name,
                    "source": str(scene_path),
                    "budget_mb": budget,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "commands": commands,
                })

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "matrix_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Prepared {len(plan)} measured runs in {args.output_root / 'matrix_plan.json'}")
    if not args.execute:
        print("Dry run only. Re-run with --execute to launch sequential GPU jobs.")
        return

    for item in plan:
        run_dir = Path(item["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        stages = ("train", "render", "metrics")
        for stage, command in zip(stages, item["commands"]):
            print(f"[{item['scene']} {item['budget_mb']}MB seed={item['seed']}] {stage}")
            return_code = run_command(command, run_dir / f"{stage}.log")
            if return_code != 0:
                print(f"Stage {stage} failed with exit code {return_code}; continuing to next run.")
                break


if __name__ == "__main__":
    main()
