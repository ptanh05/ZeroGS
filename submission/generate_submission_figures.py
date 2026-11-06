"""
Publication Figure Generator for PV-SRA-3DGS Submission Package.
Produces high-resolution 300 DPI vector PDF and PNG figures in submission/figures/.
Runs purely on CPU using matplotlib 'Agg' backend.
"""

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

def generate_fig2_convergence():
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Seeds comparison on Bonsai
    methods = ["A1_V4_P1\n(Baseline)", "ABL-HP\n(PV-SRA-3DGS)", "V_ref\n(Vanilla 3DGS)"]
    means = [20.7455, 31.8305, 29.5857]
    stds = [1.8691, 0.4789, 0.0]
    colors = ["#d95f02", "#1b9e77", "#7570b3"]
    
    bars = ax.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black", width=0.55)
    
    # Annotate values
    for bar, mean, std in zip(bars, means, stds):
        yval = bar.get_height()
        txt = f"{mean:.2f} dB" if std == 0 else f"{mean:.2f} ± {std:.2f} dB"
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.0, txt, ha="center", va="bottom", fontweight="bold", fontsize=9)
    
    # Draw paired diff arrow
    ax.annotate(
        "+11.08 dB\n(p=0.0145)",
        xy=(0.5, 24), xytext=(0.5, 28),
        ha="center", fontsize=10, fontweight="bold", color="#1b9e77",
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#1b9e77")
    )
    
    ax.set_ylabel("Novel-View Test PSNR (dB)")
    ax.set_title("Reconstruction Quality on Bonsai (30K Steps, Holdout=37)")
    ax.set_ylim(15, 36)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_convergence_psnr.png")
    fig.savefig(OUTPUT_DIR / "fig2_convergence_psnr.pdf")
    plt.close(fig)
    print("Generated fig2_convergence_psnr")

def generate_fig3_vram():
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Peak physical VRAM comparison across 3 scenes
    scenes = ["bonsai", "garden", "flowers"]
    v4_vram = [7196.0, 7226.0, 7249.0]
    hp_vram = [7337.7, 7370.0, 7390.0]
    
    x = np.arange(len(scenes))
    width = 0.35
    
    ax.bar(x - width/2, v4_vram, width, label="Baseline (A1_V4_P1)", color="#d95f02", alpha=0.85, edgecolor="black")
    ax.bar(x + width/2, hp_vram, width, label="PV-SRA-3DGS (ABL-HP)", color="#1b9e77", alpha=0.85, edgecolor="black")
    
    # Physical hardware limit
    ax.axhline(8150.56, color="red", linestyle="--", lw=1.5, label="Physical VRAM (8,151 MiB)")
    ax.axhline(7656.0, color="purple", linestyle=":", lw=1.2, label="Vanilla 3DGS Peak (7,656 MiB)")
    
    ax.set_ylabel("Peak Physical VRAM (MiB)")
    ax.set_title("Physical Peak VRAM Safety (NVIDIA RTX 5060 8GB)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Bonsai (Indoor)", "Garden (Outdoor)", "Flowers (Flora)"])
    ax.set_ylim(6000, 8500)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_vram_telemetry.png")
    fig.savefig(OUTPUT_DIR / "fig3_vram_telemetry.pdf")
    plt.close(fig)
    print("Generated fig3_vram_telemetry")

def generate_fig4_cross_scene():
    fig, ax = plt.subplots(figsize=(6, 4))
    
    scenes = ["Bonsai\n(37 Holdout)", "Garden\n(24 Holdout)", "Flowers\n(22 Holdout)"]
    v4_psnr = [20.75, 18.67, 17.36]
    hp_psnr = [31.83, 24.93, 20.30]
    gains = [hp - v4 for v4, hp in zip(v4_psnr, hp_psnr)]
    
    x = np.arange(len(scenes))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, v4_psnr, width, label="Baseline Bounded", color="#d95f02", alpha=0.85, edgecolor="black")
    rects2 = ax.bar(x + width/2, hp_psnr, width, label="PV-SRA-3DGS (Ours)", color="#1b9e77", alpha=0.85, edgecolor="black")
    
    for i, g in enumerate(gains):
        ax.text(x[i] + width/2, hp_psnr[i] + 0.6, f"+{g:.2f} dB", ha="center", va="bottom", fontweight="bold", color="#1b9e77", fontsize=9)
    
    ax.set_ylabel("Novel-View PSNR (dB)")
    ax.set_title("Cross-Scene Performance Across Diverse Topologies")
    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.set_ylim(14, 35)
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_cross_scene_comparison.png")
    fig.savefig(OUTPUT_DIR / "fig4_cross_scene_comparison.pdf")
    plt.close(fig)
    print("Generated fig4_cross_scene_comparison")

def generate_fig5_over_pruning():
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    
    # Schematic showing Opacity vs Steps for immature Gaussians
    steps = np.linspace(0, 1000, 500)
    # Sigmoidal growth curve of opacity
    opacity_normal = 0.001 + 0.08 / (1.0 + np.exp(-0.01 * (steps - 400)))
    
    ax.plot(steps, opacity_normal, color="#1b9e77", lw=2.5, label="Preservation Lifecycle (Ours, reaches convergence)")
    
    # Over-pruned sawtooth
    sawtooth_x = []
    sawtooth_y = []
    for cycle in range(5):
        s_start = cycle * 100
        s_end = (cycle + 1) * 100
        seg_x = np.linspace(s_start, s_end, 50)
        seg_y = 0.001 + 0.003 * (seg_x - s_start) / 100.0
        sawtooth_x.extend(seg_x)
        sawtooth_y.extend(seg_y)
    
    ax.plot(sawtooth_x, sawtooth_y, color="#d95f02", lw=2, linestyle="-", label="Periodic Hygiene Pruning (Thrashing loop)")
    ax.axhline(0.005, color="black", linestyle="--", lw=1.2, label=r"Pruning Threshold ($\alpha = 0.005$)")
    
    ax.annotate(
        "Culled at 100 steps\nbefore opacity matures",
        xy=(100, 0.004), xytext=(150, 0.015),
        arrowprops=dict(arrowstyle="->", color="#d95f02", lw=1.5),
        fontsize=8.5, fontweight="bold", color="#d95f02"
    )
    
    ax.annotate(
        "Matures past threshold\ninto stable geometry",
        xy=(500, 0.06), xytext=(550, 0.04),
        arrowprops=dict(arrowstyle="->", color="#1b9e77", lw=1.5),
        fontsize=8.5, fontweight="bold", color="#1b9e77"
    )
    
    ax.set_xlabel("Optimization Steps Since Primitive Spawning")
    ax.set_ylabel("Gaussian Opacity Accumulation")
    ax.set_title("The Over-Pruning Dilemma Causal Mechanism")
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 0.08)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_over_pruning_dilemma.png")
    fig.savefig(OUTPUT_DIR / "fig5_over_pruning_dilemma.pdf")
    plt.close(fig)
    print("Generated fig5_over_pruning_dilemma")

if __name__ == "__main__":
    generate_fig2_convergence()
    generate_fig3_vram()
    generate_fig4_cross_scene()
    generate_fig5_over_pruning()
    print("All submission figures generated successfully in submission/figures/")

