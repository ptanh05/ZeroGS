"""
Generate publication-quality empirical comparison charts for ZeroGS 4GB benchmark matrix.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    summary_path = "experiments/results/matrix_4gb/matrix_summary.json"
    if not os.path.exists(summary_path):
        print(f"File not found: {summary_path}")
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = list(data.keys())
    display_names = ["Vanilla\n(FP32 Adam)", "ZeroGS\n(FP32 Adam)", "ZeroGS\n(FP16 Adam)", "ZeroGS\n(INT8 Adam)"]

    psnr = [data[m]["metrics"]["PSNR"] for m in models]
    ssim = [data[m]["metrics"]["SSIM"] for m in models]
    lpips = [data[m]["metrics"]["LPIPS"] for m in models]
    peak_alloc = [data[m]["telemetry"]["peak_allocated_mb"] for m in models]
    peak_res = [data[m]["telemetry"]["peak_reserved_mb"] for m in models]
    gaussians = [data[m]["telemetry"]["final_gaussians"] / 1000.0 for m in models]
    train_speed = [data[m]["training_speed_it_per_sec"] for m in models]
    train_time = [data[m]["training_time_sec"] for m in models]
    splat_mb = [data[m]["files"]["splat_size_mb"] for m in models]
    ply_mb = [data[m]["files"]["ply_size_mb"] for m in models]

    colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ZeroGS 4GB Hard-Budget Empirical Benchmark (RTX 5060 Laptop GPU - 3,000 Steps)", fontsize=16, fontweight="bold")

    # 1. PSNR & LPIPS
    ax1 = axes[0, 0]
    bars1 = ax1.bar(display_names, psnr, color=colors, alpha=0.85, edgecolor="black", width=0.55)
    ax1.set_title("Reconstruction Quality: PSNR (dB) [Higher is Better]", fontsize=12, fontweight="bold")
    ax1.set_ylabel("PSNR (dB)", fontsize=11)
    ax1.set_ylim(12.0, 16.0)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.08, f"{yval:.2f} dB", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 2. Peak VRAM Memory
    ax2 = axes[0, 1]
    x_indices = np.arange(len(models))
    w = 0.35
    b_alloc = ax2.bar(x_indices - w/2, peak_alloc, width=w, label="Peak Allocated", color="#4e79a7", edgecolor="black")
    b_res = ax2.bar(x_indices + w/2, peak_res, width=w, label="Peak Reserved", color="#e15759", edgecolor="black")
    ax2.axhline(4096.0, color="red", linestyle="--", linewidth=2, label="Hard Budget Limit (4096 MB)")
    ax2.set_title("VRAM Consumption (MB) [Lower is Better]", fontsize=12, fontweight="bold")
    ax2.set_ylabel("VRAM (MB)", fontsize=11)
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(display_names)
    ax2.set_ylim(0, 5200)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in b_res:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 60, f"{int(yval)}M", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 3. Gaussian Count & SPLAT Size
    ax3 = axes[1, 0]
    bars3 = ax3.bar(display_names, gaussians, color=colors, alpha=0.85, edgecolor="black", width=0.55)
    ax3.set_title("Gaussian Primitives Count (x1000) & Model Size", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Gaussians (Thousands)", fontsize=11)
    ax3.set_ylim(0, 380)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    for i, bar in enumerate(bars3):
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 6, f"{yval:.1f}k\n({splat_mb[i]:.1f}MB)", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 4. Training Speed & Time
    ax4 = axes[1, 1]
    bars4 = ax4.bar(display_names, train_speed, color=colors, alpha=0.85, edgecolor="black", width=0.55)
    ax4.set_title("Training Throughput: Speed (it/s) [Higher is Better]", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Iterations per second", fontsize=11)
    ax4.set_ylim(0, 16)
    ax4.grid(axis="y", linestyle="--", alpha=0.5)
    for i, bar in enumerate(bars4):
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f} it/s\n({train_time[i]:.1f}s)", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_img = "experiments/results/matrix_4gb/matrix_comparison_chart.png"
    plt.savefig(out_img, dpi=300)
    print(f"[SUCCESS] Saved chart to {out_img}")

if __name__ == "__main__":
    main()
