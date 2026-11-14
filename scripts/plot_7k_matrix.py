"""
Generate publication-quality empirical comparison chart for ZeroGS 7k benchmark matrix.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    summary_path = "experiments/results/matrix_7k/matrix_summary.json"
    if not os.path.exists(summary_path):
        print(f"File not found: {summary_path}")
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = ["Vanilla_FP32", "ZeroGS_INT8"]
    display_names = ["Vanilla 3DGS\n(Unconstrained FP32)", "ZeroGS (Ours)\n(INT8 Adam + 4GB Cap)"]

    psnr = [data[m]["metrics"]["PSNR"] for m in models]
    ssim = [data[m]["metrics"]["SSIM"] for m in models]
    lpips = [data[m]["metrics"]["LPIPS"] for m in models]
    peak_alloc = [data[m]["telemetry"]["peak_allocated_mb"] for m in models]
    peak_res = [data[m]["telemetry"]["peak_reserved_mb"] for m in models]
    gaussians = [data[m]["telemetry"]["final_gaussians"] / 1000.0 for m in models]
    splat_mb = [data[m]["files"]["splat_size_mb"] for m in models]
    ply_mb = [data[m]["files"]["ply_size_mb"] for m in models]

    colors = ["#7f7f7f", "#2ca02c"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("ZeroGS 4GB Hard Budget vs Vanilla 3DGS (RTX 5060 Laptop GPU - 7,000 Steps)", fontsize=15, fontweight="bold")

    # 1. PSNR & Reconstruction
    ax1 = axes[0]
    bars1 = ax1.bar(display_names, psnr, color=colors, alpha=0.85, edgecolor="black", width=0.45)
    ax1.set_title("Novel View Synthesis: PSNR (dB)\n[Higher is Better]", fontsize=12, fontweight="bold")
    ax1.set_ylabel("PSNR (dB)", fontsize=11)
    ax1.set_ylim(0, 30.0)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.2f} dB", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # 2. VRAM Consumption vs 4GB Ceiling
    ax2 = axes[1]
    x_indices = np.arange(len(models))
    w = 0.3
    b_alloc = ax2.bar(x_indices - w/2, peak_alloc, width=w, label="Peak Allocated", color="#4e79a7", edgecolor="black")
    b_res = ax2.bar(x_indices + w/2, peak_res, width=w, label="Peak Reserved", color="#e15759", edgecolor="black")
    ax2.axhline(4096.0, color="red", linestyle="--", linewidth=2.5, label="Hard Budget Ceiling (4096 MB)")
    ax2.set_title("VRAM Memory Footprint (MB)\n[Lower is Better]", fontsize=12, fontweight="bold")
    ax2.set_ylabel("VRAM (MB)", fontsize=11)
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(display_names, fontsize=10)
    ax2.set_ylim(0, 5200)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in b_res:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 60, f"{int(yval)} MB", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 3. Primitives Count & File Footprint
    ax3 = axes[2]
    bars3 = ax3.bar(display_names, gaussians, color=colors, alpha=0.85, edgecolor="black", width=0.45)
    ax3.set_title("Primitive Efficiency & Disk Size\n[Lower is Better]", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Gaussians (Thousands)", fontsize=11)
    ax3.set_ylim(0, 1200)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    for i, bar in enumerate(bars3):
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 20, f"{yval:.0f}k pts\n(.ply: {ply_mb[i]:.0f}MB\n.splat: {splat_mb[i]:.1f}MB)", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out_img = "experiments/results/matrix_7k/matrix_7k_comparison_chart.png"
    plt.savefig(out_img, dpi=300)
    print(f"[SUCCESS] Saved chart to {out_img}")

if __name__ == "__main__":
    main()

