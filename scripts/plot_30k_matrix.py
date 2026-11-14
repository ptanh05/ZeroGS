"""
Generate publication-quality empirical comparison chart for ZeroGS 30k full convergence benchmark.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    summary_30k_path = "experiments/results/matrix_30k/matrix_summary.json"
    with open(summary_30k_path, "r", encoding="utf-8") as f:
        data_30k = json.load(f)

    # 30k models
    models = ["Vanilla 3DGS (FP32)\nUnbounded 8GB Baseline", "ZeroGS INT8 (Ours)\n4096MB Hard Budget Cap"]
    
    psnr = [29.5857, data_30k["ZeroGS_INT8"]["metrics"]["PSNR"]]
    ssim = [0.9276, data_30k["ZeroGS_INT8"]["metrics"]["SSIM"]]
    lpips = [0.1796, data_30k["ZeroGS_INT8"]["metrics"]["LPIPS"]]
    peak_vram = [7656.0, data_30k["ZeroGS_INT8"]["telemetry"]["peak_reserved_mb"]]
    gaussians = [1245.42, data_30k["ZeroGS_INT8"]["telemetry"]["final_gaussians"] / 1000.0]
    splat_mb = [38.0, data_30k["ZeroGS_INT8"]["files"]["splat_size_mb"]]
    ply_mb = [296.0, data_30k["ZeroGS_INT8"]["files"]["ply_size_mb"]]

    colors = ["#7f7f7f", "#2ca02c"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.suptitle("ZeroGS Full Convergence (30,000 Steps) under 4096 MB Hard Cap on RTX 5060 Laptop GPU", fontsize=14, fontweight="bold")

    # 1. PSNR & Reconstruction
    ax1 = axes[0]
    bars1 = ax1.bar(models, psnr, color=colors, alpha=0.85, edgecolor="black", width=0.45)
    ax1.set_title("Novel View Synthesis: PSNR (dB)\n[Higher is Better]", fontsize=12, fontweight="bold")
    ax1.set_ylabel("PSNR (dB)", fontsize=11)
    ax1.set_ylim(0, 35.0)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.6, f"{yval:.2f} dB", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # 2. VRAM Consumption vs 4GB Ceiling
    ax2 = axes[1]
    bars2 = ax2.bar(models, peak_vram, color=["#e15759", "#4e79a7"], alpha=0.85, edgecolor="black", width=0.45)
    ax2.axhline(4096.0, color="red", linestyle="--", linewidth=2.5, label="Hard Budget Ceiling (4096 MB)")
    ax2.set_title("Peak Reserved VRAM Footprint (MB)\n[Lower is Better]", fontsize=12, fontweight="bold")
    ax2.set_ylabel("VRAM (MB)", fontsize=11)
    ax2.set_ylim(0, 8800)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 100, f"{int(yval)} MB", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # 3. Primitives Count & Disk Footprint
    ax3 = axes[2]
    bars3 = ax3.bar(models, gaussians, color=colors, alpha=0.85, edgecolor="black", width=0.45)
    ax3.set_title("Primitives Efficiency & Model Size\n[Lower is Better]", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Gaussians (Thousands)", fontsize=11)
    ax3.set_ylim(0, 1500)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    for i, bar in enumerate(bars3):
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 25, f"{yval:.0f}k pts\n(.ply: {ply_mb[i]:.0f}MB\n.splat: {splat_mb[i]:.1f}MB)", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out_img = "experiments/results/matrix_30k/matrix_30k_comparison_chart.png"
    plt.savefig(out_img, dpi=300)
    print(f"[SUCCESS] Saved chart to {out_img}")

if __name__ == "__main__":
    main()
