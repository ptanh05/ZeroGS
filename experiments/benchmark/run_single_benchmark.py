"""
ZeroGS Single Experiment Benchmark Runner (run_single_benchmark.py).

Handles:
1. Dynamic scene resolution and holdout view count computation.
2. Pre-flight hardware and software environment verification.
3. Background hardware VRAM sampling via nvidia-smi.
4. Safe execution of train.py with collision and overwrite protection.
5. Post-training rendering at iterations 7,000 and 30,000 on dynamically determined holdout views.
6. Metric computation (PSNR, SSIM, LPIPS) matching standard evaluation protocol.
7. Validation of outputs (NaN/Inf check, PLY vertex counts vs telemetry).
8. Production of self-contained, auditable benchmark_verification.json.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from PIL import Image
import torch
import torchvision.transforms.functional as TF


EXPERIMENT_CONFIGS: Dict[str, List[str]] = {
    # ===== Reference & Baseline Configurations =====
    "V_ref": [
        "--disable_zero_oom",
    ],
    "A0": [
        "--hard_vram_limit_mb", "6144.0",
        "--safety_headroom_mb", "512.0",
        "--non_pytorch_allowance_mb", "256.0",
        "--pruning_mode", "legacy_as_is",
        "--occlusion_physics_mode", "legacy",
        "--gradient_mask_mode", "canonical",
        "--allocator_strategy", "gain_per_byte",
        "--cost_calibration", "experiments/results/cost_calibration.json",
    ],
    "A1": [
        "--hard_vram_limit_mb", "6144.0",
        "--safety_headroom_mb", "512.0",
        "--non_pytorch_allowance_mb", "256.0",
        "--pruning_mode", "decoupled",
        "--occlusion_physics_mode", "legacy",
        "--gradient_mask_mode", "canonical",
        "--allocator_strategy", "gain_per_byte",
        "--cost_calibration", "experiments/results/cost_calibration.json",
    ],
    "A2": [
        "--hard_vram_limit_mb", "6144.0",
        "--safety_headroom_mb", "512.0",
        "--non_pytorch_allowance_mb", "256.0",
        "--pruning_mode", "legacy_as_is",
        "--occlusion_physics_mode", "corrected",
        "--gradient_mask_mode", "canonical",
        "--allocator_strategy", "gain_per_byte",
        "--cost_calibration", "experiments/results/cost_calibration.json",
    ],
    "A3": [
        "--hard_vram_limit_mb", "6144.0",
        "--safety_headroom_mb", "512.0",
        "--non_pytorch_allowance_mb", "256.0",
        "--pruning_mode", "decoupled",
        "--occlusion_physics_mode", "corrected",
        "--gradient_mask_mode", "canonical",
        "--allocator_strategy", "gain_per_byte",
        "--cost_calibration", "experiments/results/cost_calibration.json",
    ],
    # ===== PV-SRA-3DGS V2 Configurations (Grace-Aware Decoupled Pruning) =====
    "A1_G": [
        "--hard_vram_limit_mb", "6144.0",
        "--safety_headroom_mb", "512.0",
        "--non_pytorch_allowance_mb", "256.0",
        "--pruning_mode", "decoupled",
        "--occlusion_physics_mode", "legacy",
        "--gradient_mask_mode", "canonical",
        "--allocator_strategy", "gain_per_byte",
        "--cost_calibration", "experiments/results/cost_calibration.json",
        "--tau_age", "500",
        "--tau_reset", "400",
        "--kappa_split", "0.40",
    ],
    "A3_G": [
        "--hard_vram_limit_mb", "6144.0",
        "--safety_headroom_mb", "512.0",
        "--non_pytorch_allowance_mb", "256.0",
        "--pruning_mode", "decoupled",
        "--occlusion_physics_mode", "corrected",
        "--gradient_mask_mode", "canonical",
        "--allocator_strategy", "gain_per_byte",
        "--cost_calibration", "experiments/results/cost_calibration.json",
        "--tau_age", "500",
        "--tau_reset", "400",
        "--kappa_split", "0.40",
    ],
    # ===== PV-SRA-3DGS V3 Configurations (Dynamic VRAM-Aware Inflow) =====
    "A1_V3": [
        "--hard_vram_limit_mb", "6144.0",
        "--safety_headroom_mb", "512.0",
        "--non_pytorch_allowance_mb", "256.0",
        "--pruning_mode", "decoupled",
        "--occlusion_physics_mode", "legacy",
        "--gradient_mask_mode", "canonical",
        "--allocator_strategy", "gain_per_byte",
        "--cost_calibration", "experiments/results/cost_calibration.json",
        "--tau_age", "500",
        "--tau_reset", "400",
        "--kappa_split", "0.40",
        "--inflow_mode", "dynamic",
        "--inflow_cap_min", "5000",
        "--inflow_cap_max", "15000",
        "--target_headroom_mb", "1024.0",
        "--max_gaussians_cap", "1000000",
    ],
    "A3_V3": [
        "--hard_vram_limit_mb", "6144.0",
        "--safety_headroom_mb", "512.0",
        "--non_pytorch_allowance_mb", "256.0",
        "--pruning_mode", "decoupled",
        "--occlusion_physics_mode", "corrected",
        "--gradient_mask_mode", "canonical",
        "--allocator_strategy", "gain_per_byte",
        "--cost_calibration", "experiments/results/cost_calibration.json",
        "--tau_age", "500",
        "--tau_reset", "400",
        "--kappa_split", "0.40",
        "--inflow_mode", "dynamic",
        "--inflow_cap_min", "5000",
        "--inflow_cap_max", "15000",
        "--target_headroom_mb", "1024.0",
        "--max_gaussians_cap", "1000000",
    ],
    # ===== PV-SRA-3DGS V4 Configurations (Controlled Population-Cap Ablations) =====
    "A1_V4_P1": [
        "--hard_vram_limit_mb", "6144.0",
        "--safety_headroom_mb", "512.0",
        "--non_pytorch_allowance_mb", "256.0",
        "--pruning_mode", "decoupled",
        "--occlusion_physics_mode", "legacy",
        "--gradient_mask_mode", "canonical",
        "--allocator_strategy", "gain_per_byte",
        "--cost_calibration", "experiments/results/cost_calibration.json",
        "--tau_age", "500",
        "--tau_reset", "400",
        "--kappa_split", "0.40",
        "--inflow_mode", "dynamic",
        "--inflow_cap_min", "5000",
        "--inflow_cap_max", "15000",
        "--target_headroom_mb", "1024.0",
        "--max_gaussians_cap", "1150000",
    ],
    # Confirmatory Population-Cap Sweep Variants
    "C1100": [
        "--hard_vram_limit_mb", "6144.0",
        "--safety_headroom_mb", "512.0",
        "--non_pytorch_allowance_mb", "256.0",
        "--pruning_mode", "decoupled",
        "--occlusion_physics_mode", "legacy",
        "--gradient_mask_mode", "canonical",
        "--allocator_strategy", "gain_per_byte",
        "--cost_calibration", "experiments/results/cost_calibration.json",
        "--tau_age", "500",
        "--tau_reset", "400",
        "--kappa_split", "0.40",
        "--inflow_mode", "dynamic",
        "--inflow_cap_min", "5000",
        "--inflow_cap_max", "15000",
        "--target_headroom_mb", "1024.0",
        "--max_gaussians_cap", "1100000",
    ],
    "C1200": [
        "--hard_vram_limit_mb", "6144.0",
        "--safety_headroom_mb", "512.0",
        "--non_pytorch_allowance_mb", "256.0",
        "--pruning_mode", "decoupled",
        "--occlusion_physics_mode", "legacy",
        "--gradient_mask_mode", "canonical",
        "--allocator_strategy", "gain_per_byte",
        "--cost_calibration", "experiments/results/cost_calibration.json",
        "--tau_age", "500",
        "--tau_reset", "400",
        "--kappa_split", "0.40",
        "--inflow_mode", "dynamic",
        "--inflow_cap_min", "5000",
        "--inflow_cap_max", "15000",
        "--target_headroom_mb", "1024.0",
        "--max_gaussians_cap", "1200000",
    ],
    "C1300": [
        "--hard_vram_limit_mb", "6144.0",
        "--safety_headroom_mb", "512.0",
        "--non_pytorch_allowance_mb", "256.0",
        "--pruning_mode", "decoupled",
        "--occlusion_physics_mode", "legacy",
        "--gradient_mask_mode", "canonical",
        "--allocator_strategy", "gain_per_byte",
        "--cost_calibration", "experiments/results/cost_calibration.json",
        "--tau_age", "500",
        "--tau_reset", "400",
        "--kappa_split", "0.40",
        "--inflow_mode", "dynamic",
        "--inflow_cap_min", "5000",
        "--inflow_cap_max", "15000",
        "--target_headroom_mb", "1024.0",
        "--max_gaussians_cap", "1300000",
    ],
    # Controlled V4 Component Ablations
    "ABL-HP": [
        "--hard_vram_limit_mb", "6144.0",
        "--safety_headroom_mb", "512.0",
        "--non_pytorch_allowance_mb", "256.0",
        "--pruning_mode", "legacy_as_is",  # Disables Stage 1 decoupled hygiene pruning
        "--occlusion_physics_mode", "legacy",
        "--gradient_mask_mode", "canonical",
        "--allocator_strategy", "gain_per_byte",
        "--cost_calibration", "experiments/results/cost_calibration.json",
        "--tau_age", "500",
        "--tau_reset", "400",
        "--kappa_split", "0.40",
        "--inflow_mode", "dynamic",
        "--inflow_cap_min", "5000",
        "--inflow_cap_max", "15000",
        "--target_headroom_mb", "1024.0",
        "--max_gaussians_cap", "1150000",
    ],
    "ABL-DI": [
        "--hard_vram_limit_mb", "6144.0",
        "--safety_headroom_mb", "512.0",
        "--non_pytorch_allowance_mb", "256.0",
        "--pruning_mode", "decoupled",
        "--occlusion_physics_mode", "legacy",
        "--gradient_mask_mode", "canonical",
        "--allocator_strategy", "gain_per_byte",
        "--cost_calibration", "experiments/results/cost_calibration.json",
        "--tau_age", "500",
        "--tau_reset", "400",
        "--kappa_split", "0.40",
        "--inflow_mode", "static",
        "--gaussians_per_iter_cap", "5000",
        "--max_gaussians_cap", "1150000",
    ],
    "ABL-DI-HP": [
        "--hard_vram_limit_mb", "6144.0",
        "--safety_headroom_mb", "512.0",
        "--non_pytorch_allowance_mb", "256.0",
        "--pruning_mode", "legacy_as_is",
        "--occlusion_physics_mode", "legacy",
        "--gradient_mask_mode", "canonical",
        "--allocator_strategy", "gain_per_byte",
        "--cost_calibration", "experiments/results/cost_calibration.json",
        "--tau_age", "500",
        "--tau_reset", "400",
        "--kappa_split", "0.40",
        "--inflow_mode", "static",
        "--gaussians_per_iter_cap", "5000",
        "--max_gaussians_cap", "1150000",
    ],
}

# Aliases with dashes and alternate naming
EXPERIMENT_CONFIGS["A1-G"] = EXPERIMENT_CONFIGS["A1_G"]
EXPERIMENT_CONFIGS["A3-G"] = EXPERIMENT_CONFIGS["A3_G"]
EXPERIMENT_CONFIGS["A1-V3"] = EXPERIMENT_CONFIGS["A1_V3"]
EXPERIMENT_CONFIGS["A3-V3"] = EXPERIMENT_CONFIGS["A3_V3"]
EXPERIMENT_CONFIGS["A1-V4-P1"] = EXPERIMENT_CONFIGS["A1_V4_P1"]
EXPERIMENT_CONFIGS["C-1100"] = EXPERIMENT_CONFIGS["C1100"]
EXPERIMENT_CONFIGS["C-1200"] = EXPERIMENT_CONFIGS["C1200"]
EXPERIMENT_CONFIGS["C-1300"] = EXPERIMENT_CONFIGS["C1300"]
EXPERIMENT_CONFIGS["ABL_HP"] = EXPERIMENT_CONFIGS["ABL-HP"]
EXPERIMENT_CONFIGS["ABL_DI"] = EXPERIMENT_CONFIGS["ABL-DI"]
EXPERIMENT_CONFIGS["ABL_DI_HP"] = EXPERIMENT_CONFIGS["ABL-DI-HP"]
EXPERIMENT_CONFIGS["ABL-DI-HP"] = EXPERIMENT_CONFIGS["ABL-DI-HP"]


class DeviceMemoryMonitor:
    """Polls nvidia-smi every interval_sec to track true peak physical memory."""

    def __init__(self, interval_sec: float = 2.0):
        self.interval_sec = interval_sec
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.peak_device_memory_mb: float = 0.0

    def _poll(self):
        while not self.stop_event.is_set():
            try:
                res = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    stderr=subprocess.DEVNULL,
                ).decode().strip()
                val = float(res.splitlines()[0])
                if val > self.peak_device_memory_mb:
                    self.peak_device_memory_mb = val
            except Exception:
                pass
            time.sleep(self.interval_sec)

    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def stop(self) -> float:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3.0)
        return self.peak_device_memory_mb


def get_git_info() -> Dict[str, Any]:
    commit = None
    dirty = False
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(status)
    except Exception:
        pass
    return {"commit": commit, "dirty": dirty}


def get_hardware_and_software_env() -> Dict[str, Any]:
    gpu_name = None
    total_vram_mb = None
    driver_version = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 2)
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version,gpu_name,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip().splitlines()[0]
        parts = [p.strip() for p in smi_out.split(",")]
        if len(parts) >= 1:
            driver_version = parts[0]
        if len(parts) >= 2 and not gpu_name:
            gpu_name = parts[1]
        if len(parts) >= 3 and not total_vram_mb:
            total_vram_mb = float(parts[2])
    except Exception:
        pass

    return {
        "os": sys.platform,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda if hasattr(torch.version, "cuda") else None,
        "gpu_name": gpu_name,
        "total_vram_mb": total_vram_mb,
        "driver_version": driver_version,
    }


def resolve_scene_and_holdout_count(
    repo_root: Path,
    scene_arg: str,
    llffhold: int = 8,
) -> Tuple[Path, int, str]:
    """
    Resolves dataset path and dynamically computes the expected number of holdout evaluation views.
    Supports simple names ('bonsai', 'garden', 'flowers', 'lego') or custom relative/absolute paths.
    """
    candidate = Path(scene_arg)
    if not candidate.is_absolute():
        if (repo_root / "data" / scene_arg).exists():
            candidate = repo_root / "data" / scene_arg
        elif (repo_root / scene_arg).exists():
            candidate = repo_root / scene_arg

    if not candidate.exists():
        raise FileNotFoundError(
            f"Scene dataset not found at '{scene_arg}' or '{repo_root / 'data' / scene_arg}'."
        )

    resolved_path = candidate.resolve()
    scene_name = resolved_path.name

    test_count: Optional[int] = None

    # 1. Check COLMAP structure
    sparse_path = resolved_path / "sparse/0"
    if not sparse_path.exists():
        sparse_path = resolved_path / "sparse"

    if sparse_path.exists():
        images_bin = sparse_path / "images.bin"
        images_txt = sparse_path / "images.txt"
        test_txt = sparse_path / "test.txt"

        if test_txt.exists():
            with open(test_txt, "r", encoding="utf-8") as f:
                test_count = len([line.strip() for line in f if line.strip()])
        elif images_bin.exists():
            from scene.colmap_loader import read_extrinsics_binary
            cams = read_extrinsics_binary(str(images_bin))
            cam_names = sorted([cam.name for cam in cams.values()])
            test_count = len([name for idx, name in enumerate(cam_names) if idx % llffhold == 0])
        elif images_txt.exists():
            from scene.colmap_loader import read_extrinsics_text
            cams = read_extrinsics_text(str(images_txt))
            cam_names = sorted([cam.name for cam in cams.values()])
            test_count = len([name for idx, name in enumerate(cam_names) if idx % llffhold == 0])

    # 2. Check Blender synthetic structure
    elif (resolved_path / "transforms_test.json").exists():
        with open(resolved_path / "transforms_test.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            test_count = len(data.get("frames", []))

    if test_count is None or test_count == 0:
        raise ValueError(
            f"Could not determine holdout evaluation view count for scene at '{resolved_path}'. "
            "Ensure the directory contains COLMAP sparse/0 or transforms_test.json."
        )

    return resolved_path, test_count, scene_name


def resolve_output_directory(
    repo_root: Path,
    exp_name: str,
    scene_name: str,
    seed: int,
    max_gaussians_cap: Optional[int],
    custom_output_dir: Optional[str] = None,
    force_overwrite: bool = False,
) -> Path:
    """
    Generates a unique, non-colliding output directory path while preserving
    strict backward compatibility for historical baseline runs on bonsai.
    """
    if custom_output_dir:
        target_dir = Path(custom_output_dir).resolve()
    else:
        # Determine default cap for this experiment to know if cap is custom
        default_cap = None
        if exp_name in EXPERIMENT_CONFIGS:
            cfg = EXPERIMENT_CONFIGS[exp_name]
            if "--max_gaussians_cap" in cfg:
                idx = cfg.index("--max_gaussians_cap")
                default_cap = int(cfg[idx + 1])

        cap_is_custom = (max_gaussians_cap is not None and max_gaussians_cap != default_cap)
        is_default_bonsai_s0 = (scene_name == "bonsai" and seed == 0 and not cap_is_custom)

        if is_default_bonsai_s0:
            target_dir = repo_root / f"output/benchmark_bonsai_{exp_name}"
        else:
            folder_parts = [f"benchmark_{scene_name}_{exp_name}"]
            if seed != 0:
                folder_parts.append(f"s{seed}")
            if cap_is_custom:
                folder_parts.append(f"c{max_gaussians_cap}")
            target_dir = repo_root / f"output/{'_'.join(folder_parts)}"

    # Collision Prevention
    verif_file = target_dir / "benchmark_verification.json"
    ckpt_file = target_dir / "chkpnt30000.pth"
    if (verif_file.exists() or ckpt_file.exists()) and not force_overwrite:
        raise FileExistsError(
            f"Benchmark output directory '{target_dir}' already contains completed run artifacts. "
            f"Aborting execution to protect existing data integrity. "
            f"Use --force_overwrite to rerun or specify a unique --output_dir / --seed."
        )

    return target_dir


def build_experiment_args(
    exp_name: str,
    max_gaussians_cap: Optional[int] = None,
    hard_vram_limit_mb: Optional[float] = None,
    safety_headroom_mb: Optional[float] = None,
    non_pytorch_allowance_mb: Optional[float] = None,
    target_headroom_mb: Optional[float] = None,
) -> List[str]:
    """Builds and overrides experiment argument flags."""
    args_list = list(EXPERIMENT_CONFIGS[exp_name])

    def set_or_replace(flag: str, val: Any):
        if flag in args_list:
            idx = args_list.index(flag)
            args_list[idx + 1] = str(val)
        else:
            args_list.extend([flag, str(val)])

    if max_gaussians_cap is not None:
        set_or_replace("--max_gaussians_cap", max_gaussians_cap)
    if hard_vram_limit_mb is not None:
        set_or_replace("--hard_vram_limit_mb", hard_vram_limit_mb)
    if safety_headroom_mb is not None:
        set_or_replace("--safety_headroom_mb", safety_headroom_mb)
    if non_pytorch_allowance_mb is not None:
        set_or_replace("--non_pytorch_allowance_mb", non_pytorch_allowance_mb)
    if target_headroom_mb is not None:
        set_or_replace("--target_headroom_mb", target_headroom_mb)

    return args_list


def read_ply_vertex_count(ply_path: str) -> Optional[int]:
    if not os.path.exists(ply_path):
        return None
    try:
        with open(ply_path, "rb") as f:
            for _ in range(25):
                line = f.readline().decode("ascii", errors="ignore").strip()
                if line.startswith("element vertex"):
                    return int(line.split()[-1])
                if line == "end_header":
                    break
    except Exception:
        pass
    return None


def verify_images(renders_dir: str, gts_dir: str, expected_count: int) -> Dict[str, Any]:
    if not os.path.exists(renders_dir) or not os.path.exists(gts_dir):
        return {"valid": False, "error": "Directories missing", "count": 0}
    r_files = sorted([f for f in os.listdir(renders_dir) if f.lower().endswith(".png")])
    g_files = sorted([f for f in os.listdir(gts_dir) if f.lower().endswith(".png")])
    if len(r_files) != expected_count or len(g_files) != expected_count:
        return {
            "valid": False,
            "error": f"Count mismatch: renders={len(r_files)}, gts={len(g_files)}, expected={expected_count}",
            "count": len(r_files),
        }

    nan_inf_found = False
    out_of_range = False
    for rf in r_files:
        img = TF.to_tensor(Image.open(os.path.join(renders_dir, rf)))
        if torch.isnan(img).any() or torch.isinf(img).any():
            nan_inf_found = True
        if (img < -1e-4).any() or (img > 1.0001).any():
            out_of_range = True

    return {
        "valid": not nan_inf_found and not out_of_range,
        "count": len(r_files),
        "expected_count": expected_count,
        "nan_inf_found": nan_inf_found,
        "out_of_range": out_of_range,
    }


def main():
    parser = argparse.ArgumentParser(description="ZeroGS Dynamic Multi-Scene Benchmark Runner")
    parser.add_argument(
        "--experiment",
        choices=list(EXPERIMENT_CONFIGS.keys()),
        required=True,
        help="Experiment architecture configuration key",
    )
    parser.add_argument("--scene", type=str, default="bonsai", help="Scene name (in data/) or path")
    parser.add_argument("--iterations", type=int, default=30_000, help="Training iterations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--images", type=str, default="images_4", help="Image directory name (e.g. images_4)")
    parser.add_argument("--max_gaussians_cap", type=int, default=None, help="Explicit population cap override")
    parser.add_argument("--hard_vram_limit_mb", type=float, default=None, help="VRAM limit override (MB)")
    parser.add_argument("--safety_headroom_mb", type=float, default=None, help="Safety headroom override (MB)")
    parser.add_argument("--non_pytorch_allowance_mb", type=float, default=None, help="Non-PyTorch allowance (MB)")
    parser.add_argument("--target_headroom_mb", type=float, default=None, help="Target headroom override (MB)")
    parser.add_argument("--output_dir", type=str, default=None, help="Explicit target directory override")
    parser.add_argument("--force_overwrite", action="store_true", help="Allow overwriting completed experiments")
    parser.add_argument("--repo_root", type=str, default=".", help="ZeroGS root directory")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and execute evaluation pipeline")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    exp_name = args.experiment

    # 1. Resolve Scene and Dynamic Evaluation View Count
    scene_path, expected_holdout_views, scene_name = resolve_scene_and_holdout_count(
        repo_root=repo_root,
        scene_arg=args.scene,
        llffhold=8,
    )

    # 2. Resolve Unique Non-Colliding Output Directory
    model_path = resolve_output_directory(
        repo_root=repo_root,
        exp_name=exp_name,
        scene_name=scene_name,
        seed=args.seed,
        max_gaussians_cap=args.max_gaussians_cap,
        custom_output_dir=args.output_dir,
        force_overwrite=args.force_overwrite,
    )

    telemetry_path = model_path / "training_telemetry.jsonl"
    log_path = model_path / "run.log"

    # 3. Build Effective Experiment Configuration Flags
    effective_config = build_experiment_args(
        exp_name=exp_name,
        max_gaussians_cap=args.max_gaussians_cap,
        hard_vram_limit_mb=args.hard_vram_limit_mb,
        safety_headroom_mb=args.safety_headroom_mb,
        non_pytorch_allowance_mb=args.non_pytorch_allowance_mb,
        target_headroom_mb=args.target_headroom_mb,
    )

    print(f"\n=======================================================")
    print(f" PV-SRA-3DGS BENCHMARK EXECUTION: {exp_name}")
    print(f" Scene: {scene_name} ({scene_path})")
    print(f" Expected Holdout Evaluation Views: {expected_holdout_views}")
    print(f" Target Directory: {model_path}")
    print(f" Iterations: {args.iterations}")
    print(f" Seed: {args.seed}")
    print(f" Effective Configuration Flags: {effective_config}")
    print(f"=======================================================\n")

    os.makedirs(model_path, exist_ok=True)

    # 4. Construct train.py command
    python_exe = sys.executable
    train_cmd = [
        python_exe, str(repo_root / "train.py"),
        "-s", str(scene_path),
        "-i", args.images,
        "-m", str(model_path),
        "--eval",
        "--iterations", str(args.iterations),
        "--save_iterations", "7000", str(args.iterations),
        "--test_iterations", "7000", str(args.iterations),
        "--checkpoint_iterations", "7000", str(args.iterations),
        "--seed", str(args.seed),
        "--experiment_method", exp_name,
        "--experiment_scene", scene_name,
        "--telemetry_jsonl", str(telemetry_path),
        *effective_config,
    ]

    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"

    t_start = time.time()
    return_code = 0
    peak_device_mem = 0.0

    if not args.eval_only:
        print(f"[BENCHMARK] Launching train.py:")
        print(" ".join(train_cmd))

        monitor = DeviceMemoryMonitor(interval_sec=1.5)
        monitor.start()

        with open(log_path, "w", encoding="utf-8") as log_f:
            process = subprocess.Popen(
                train_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=str(repo_root),
                env=child_env,
            )
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_f.write(line)
                log_f.flush()
            return_code = process.wait()

        peak_device_mem = monitor.stop()
        print(f"\n[BENCHMARK] Training finished with return code {return_code}.")
    else:
        print(f"[BENCHMARK] Running in --eval_only mode (skipping training).")

    t_end = time.time()
    duration_sec = t_end - t_start

    # 5. Read run_status.json
    status_file = model_path / "run_status.json"
    status_data = {}
    if status_file.exists():
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                status_data = json.load(f)
        except Exception:
            pass

    # 6. Read final telemetry from jsonl
    telemetry_summary = {}
    if telemetry_path.exists():
        try:
            with open(telemetry_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            for l in reversed(lines):
                obj = json.loads(l)
                if obj.get("event") == "training_end":
                    telemetry_summary = obj
                    break
        except Exception:
            pass

    final_ply_path = model_path / f"point_cloud/iteration_{args.iterations}/point_cloud.ply"
    ply_vertex_count = read_ply_vertex_count(str(final_ply_path))

    # 7. Post-Training Evaluation Pipeline
    eval_7k = None
    eval_30k = None
    img_val_7k = None
    img_val_30k = None

    if return_code == 0 and (args.eval_only or status_data.get("status") == "completed"):
        print(f"\n[BENCHMARK] Beginning evaluation on {expected_holdout_views} holdout views...")

        # Render 7K
        print(f"[BENCHMARK] Rendering iteration 7000...")
        render_7k_cmd = [
            python_exe, str(repo_root / "render.py"),
            "-s", str(scene_path),
            "-i", args.images,
            "-m", str(model_path),
            "--eval", "--skip_train",
            "--iteration", "7000",
        ]
        subprocess.run(render_7k_cmd, cwd=str(repo_root), check=True, env=child_env)

        # Render 30K
        print(f"[BENCHMARK] Rendering iteration {args.iterations}...")
        render_30k_cmd = [
            python_exe, str(repo_root / "render.py"),
            "-s", str(scene_path),
            "-i", args.images,
            "-m", str(model_path),
            "--eval", "--skip_train",
            "--iteration", str(args.iterations),
        ]
        subprocess.run(render_30k_cmd, cwd=str(repo_root), check=True, env=child_env)

        # Metrics computation
        metrics_cmd = [
            python_exe, str(repo_root / "metrics.py"),
            "-m", str(model_path),
            "-o", str(model_path / "metrics_summary.json"),
        ]
        subprocess.run(metrics_cmd, cwd=str(repo_root), check=True, env=child_env)

        # Image Validation with dynamic expected count
        img_val_7k = verify_images(
            str(model_path / "test/ours_7000/renders"),
            str(model_path / "test/ours_7000/gt"),
            expected_count=expected_holdout_views,
        )
        img_val_30k = verify_images(
            str(model_path / f"test/ours_{args.iterations}/renders"),
            str(model_path / f"test/ours_{args.iterations}/gt"),
            expected_count=expected_holdout_views,
        )

        # Load metrics json
        metrics_file = model_path / "metrics_summary.json"
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                all_metrics = json.load(f)
                for k, v in all_metrics.items():
                    if "7000" in k:
                        eval_7k = v
                    elif str(args.iterations) in k:
                        eval_30k = v

    # 8. Build consolidated verification record
    verification_data = {
        "experiment": exp_name,
        "scene": scene_name,
        "scene_path": str(scene_path),
        "expected_eval_views": expected_holdout_views,
        "return_code": return_code,
        "training_status": status_data.get("status", "unknown" if not args.eval_only else "completed"),
        "oom": status_data.get("oom", False),
        "budget_violation": status_data.get("budget_exceeded", False),
        "duration_seconds": round(duration_sec, 2),
        "throughput_it_per_sec": round(args.iterations / max(1.0, duration_sec), 2),
        "peak_device_memory_mb": round(peak_device_mem, 2),
        "ply_final_gaussians": ply_vertex_count,
        "telemetry_final_gaussians": telemetry_summary.get("final_gaussians", None),
        "telemetry_memory": telemetry_summary.get("memory", {}),
        "eval_7k": eval_7k,
        "eval_30k": eval_30k,
        "image_validation_7k": img_val_7k,
        "image_validation_30k": img_val_30k,
        "git": get_git_info(),
        "environment": get_hardware_and_software_env(),
        "effective_configuration": {
            "experiment": exp_name,
            "scene": scene_name,
            "seed": args.seed,
            "iterations": args.iterations,
            "max_gaussians_cap_override": args.max_gaussians_cap,
            "command_line_flags": effective_config,
        },
    }

    verif_path = model_path / "benchmark_verification.json"
    with open(verif_path, "w", encoding="utf-8") as vf:
        json.dump(verification_data, vf, indent=2, default=str)

    print(f"\n[BENCHMARK] Verification record saved to: {verif_path}")
    print(json.dumps(verification_data, indent=2, default=str))

    if return_code != 0 or (not args.eval_only and status_data.get("status") != "completed"):
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
