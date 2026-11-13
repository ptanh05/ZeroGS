"""
ZeroGS: Batch export all trained 3DGS models to compact WebGL .splat and compressed .npz.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scene.gaussian_model import GaussianModel

TARGET_RUNS = [
    "benchmark_bonsai_ABL-HP",
    "benchmark_bonsai_A1_V4_P1",
    "benchmark_bonsai_V_ref",
    "benchmark_garden_ABL-HP",
    "benchmark_garden_A1_V4_P1",
    "benchmark_flowers_ABL-HP",
    "benchmark_flowers_A1_V4_P1",
]


def main():
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "output"

    print("=" * 70)
    print("ZeroGS: Batch Model Compaction Pipeline (.splat & .npz)")
    print("=" * 70)

    for run_name in TARGET_RUNS:
        run_path = output_dir / run_name / "point_cloud" / "iteration_30000"
        ply_file = run_path / "point_cloud.ply"
        if not ply_file.exists():
            print(f"[-] Skipping {run_name} (ply not found)")
            continue

        splat_file = run_path / "point_cloud.splat"
        npz_file = run_path / "point_cloud_compressed.npz"

        orig_size_mb = ply_file.stat().st_size / (1024 * 1024)

        gm = GaussianModel(sh_degree=3)
        gm.load_ply(str(ply_file))
        n_gaussians = gm.get_xyz.shape[0]

        gm.save_splat(str(splat_file))
        splat_size_mb = splat_file.stat().st_size / (1024 * 1024)

        gm.save_compressed(str(npz_file))
        npz_size_mb = npz_file.stat().st_size / (1024 * 1024)

        ratio = orig_size_mb / max(0.01, splat_size_mb)
        print(f"[+] {run_name:<30} | {n_gaussians:>9,} pts | PLY: {orig_size_mb:>6.1f}MB -> SPLAT: {splat_size_mb:>5.1f}MB ({ratio:.1f}x) | NPZ: {npz_size_mb:>5.1f}MB")

    print("=" * 70)
    print("All models successfully exported to compact formats.")


if __name__ == "__main__":
    main()
