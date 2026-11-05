import json
import csv
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(".").resolve()

# 1. Gather all verification records
benchmarks = ["V_ref", "A0", "A1", "A2", "A3", "A1_G", "A3_G", "A1_V3", "A3_V3"]
records = {}

for exp in benchmarks:
    verif_path = repo_root / f"output/benchmark_bonsai_{exp}/benchmark_verification.json"
    if verif_path.exists():
        with open(verif_path, "r", encoding="utf-8") as f:
            records[exp] = json.load(f)
    else:
        print(f"Warning: verification record for {exp} not found at {verif_path}")

# 2. Extract detailed metrics for CSV
csv_rows = []
for exp in benchmarks:
    d = records.get(exp, {})
    eval_7k = d.get("eval_7k", {})
    eval_30k = d.get("eval_30k", {})
    mem = d.get("telemetry_memory", {})
    
    csv_rows.append({
        "Experiment": exp,
        "Method": exp,
        "7K_PSNR_dB": round(eval_7k.get("PSNR", 0.0), 4),
        "7K_SSIM": round(eval_7k.get("SSIM", 0.0), 4),
        "7K_LPIPS": round(eval_7k.get("LPIPS", 0.0), 4),
        "30K_PSNR_dB": round(eval_30k.get("PSNR", 0.0), 4),
        "30K_SSIM": round(eval_30k.get("SSIM", 0.0), 4),
        "30K_LPIPS": round(eval_30k.get("LPIPS", 0.0), 4),
        "Final_Gaussians": d.get("ply_final_gaussians", 0),
        "Peak_Allocated_MB": round(mem.get("peak_allocated_mb", 0.0), 2),
        "Peak_Reserved_MB": round(mem.get("peak_reserved_mb", 0.0), 2),
        "Peak_Device_VRAM_MB": round(d.get("peak_device_memory_mb", 0.0), 2),
        "Conservative_Footprint_MB": round(mem.get("conservative_footprint_mb", 0.0), 2),
        "Duration_Seconds": round(d.get("duration_seconds", 0.0), 2),
        "Throughput_it_per_sec": round(d.get("throughput_it_per_sec", 0.0), 2),
        "OOM": d.get("oom", False),
        "Budget_Violation": d.get("budget_violation", False),
        "Return_Code": d.get("return_code", -1),
    })

csv_path = repo_root / "experiments/results/v3_benchmark_comparison.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"Saved comparison CSV to: {csv_path}")

# 3. Telemetry deep-dive for V3 runs
def parse_telemetry_deep(telemetry_path):
    records = []
    with open(telemetry_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    mutation_cycles = [r for r in records if r.get("event") == "mutation_cycle"]
    reclaims = [r for r in records if "reclaimed_mb" in r]
    
    iterations = [r.get("iteration") for r in mutation_cycles]
    populations = [r.get("population_after") for r in mutation_cycles]
    caps = [r.get("computed_dynamic_cap") for r in mutation_cycles]
    headrooms = [r.get("available_headroom_mb") for r in mutation_cycles]
    allocs = [r.get("memory_after", {}).get("allocated_mb", 0.0) for r in mutation_cycles]
    reserveds = [r.get("memory_after", {}).get("reserved_mb", 0.0) for r in mutation_cycles]
    requested = [r.get("requested_candidates", 0) for r in mutation_cycles]
    accepted = [r.get("accepted_candidates", 0) for r in mutation_cycles]
    rej_cap = [r.get("rejected_by_cap", 0) for r in mutation_cycles]
    rej_budget = [r.get("rejected_by_memory_budget", 0) for r in mutation_cycles]
    clones = [r.get("n_clone_executed", 0) for r in mutation_cycles]
    splits = [r.get("n_split_executed", 0) for r in mutation_cycles]
    pre_prunes = [r.get("executed_pre_prune", 0) for r in mutation_cycles]
    hyg_prunes = [r.get("executed_hygiene_prune", 0) for r in mutation_cycles]
    age_prot = [r.get("n_protected_by_age", 0) for r in mutation_cycles]
    reset_prot = [r.get("n_protected_by_cooldown", 0) for r in mutation_cycles]
    
    return {
        "iterations": iterations,
        "populations": populations,
        "caps": caps,
        "headrooms": headrooms,
        "allocs": allocs,
        "reserveds": reserveds,
        "requested": requested,
        "accepted": accepted,
        "rej_cap": rej_cap,
        "rej_budget": rej_budget,
        "clones": clones,
        "splits": splits,
        "pre_prunes": pre_prunes,
        "hyg_prunes": hyg_prunes,
        "age_prot": age_prot,
        "reset_prot": reset_prot,
        "reclaims": reclaims,
        "total_requested": sum(requested),
        "total_accepted": sum(accepted),
        "total_rej_cap": sum(rej_cap),
        "total_rej_budget": sum(rej_budget),
        "total_clones": sum(clones),
        "total_splits": sum(splits),
        "total_pre_prunes": sum(pre_prunes),
        "total_hyg_prunes": sum(hyg_prunes),
        "total_age_prot": sum(age_prot),
        "total_reset_prot": sum(reset_prot),
    }

t_a1 = parse_telemetry_deep(repo_root / "output/benchmark_bonsai_A1_V3/training_telemetry.jsonl")
t_a3 = parse_telemetry_deep(repo_root / "output/benchmark_bonsai_A3_V3/training_telemetry.jsonl")

# 4. Generate High-Quality Comparison Plots
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axs = plt.subplots(2, 3, figsize=(18, 11), dpi=150)

# (1) PSNR vs Experiment
ax = axs[0, 0]
exp_labels = ["V_ref", "A0", "A1", "A1_G", "A1_V3", "A2", "A3", "A3_G", "A3_V3"]
colors = ["#4a5568", "#2b6cb0", "#c53030", "#dd6b20", "#319795", "#2b6cb0", "#c53030", "#dd6b20", "#319795"]
psnr_vals = [records[e]["eval_30k"]["PSNR"] for e in exp_labels]
bars = ax.bar(exp_labels, psnr_vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
ax.set_title("30K PSNR (dB) Across Evolutions", fontsize=12, fontweight="bold")
ax.set_ylabel("PSNR (dB)")
ax.set_ylim(15, 34)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax.tick_params(axis="x", rotation=30)

# (2) Population Evolution (V3 A1 vs A3)
ax = axs[0, 1]
ax.plot(t_a1["iterations"], t_a1["populations"], label="A1_V3 (Legacy Occ)", color="#319795", linewidth=2.0)
ax.plot(t_a3["iterations"], t_a3["populations"], label="A3_V3 (Corrected Occ)", color="#805ad5", linewidth=2.0, linestyle="--")
ax.axhline(1_000_000, color="red", linestyle=":", label="Max Population Cap (1M)")
ax.axhline(341_157, color="#dd6b20", linestyle=":", label="A1_G V2 Final Pop (341K)")
ax.set_title("Gaussian Population Dynamics (V3)", fontsize=12, fontweight="bold")
ax.set_xlabel("Iteration")
ax.set_ylabel("Primitives")
ax.legend(fontsize=8, loc="lower right")

# (3) Dynamic Inflow Cap vs Headroom
ax = axs[0, 2]
ax.plot(t_a1["iterations"], t_a1["caps"], label="Dynamic Cap (A1_V3)", color="#319795", linewidth=1.8)
ax.plot(t_a3["iterations"], t_a3["caps"], label="Dynamic Cap (A3_V3)", color="#805ad5", linewidth=1.8, linestyle="--")
ax.axhline(5000, color="grey", linestyle=":", label="V2 Static Cap (5,000)")
ax.axhline(15000, color="green", linestyle=":", label="V3 Max Cap (15,000)")
ax.set_title("Dynamic Inflow Cap Scaling", fontsize=12, fontweight="bold")
ax.set_xlabel("Iteration")
ax.set_ylabel("Cap per Cycle")
ax.set_ylim(0, 17000)
ax.legend(fontsize=8, loc="lower left")

# (4) Memory Footprint & Safety Budget
ax = axs[1, 0]
alloc_vals = [records[e]["telemetry_memory"]["peak_allocated_mb"] for e in exp_labels]
res_vals = [records[e]["telemetry_memory"]["peak_reserved_mb"] for e in exp_labels]
x = np.arange(len(exp_labels))
width = 0.35
ax.bar(x - width/2, alloc_vals, width, label="Peak Allocated", color="#4299e1", edgecolor="black", linewidth=0.8)
ax.bar(x + width/2, res_vals, width, label="Peak Reserved", color="#ed8936", edgecolor="black", linewidth=0.8)
ax.axhline(6144, color="crimson", linestyle="--", linewidth=1.5, label="Hard Budget (6,144 MB)")
ax.set_title("Peak VRAM Consumption vs Budget", fontsize=12, fontweight="bold")
ax.set_ylabel("Memory (MB)")
ax.set_xticks(x)
ax.set_xticklabels(exp_labels, rotation=30)
ax.set_ylim(0, 7000)
ax.legend(fontsize=8, loc="upper right")

# (5) Candidates Admitted vs Rejected
ax = axs[1, 1]
categories = ["A1_V3", "A3_V3"]
admitted = [t_a1["total_accepted"], t_a3["total_accepted"]]
rejected_cap = [t_a1["total_rej_cap"], t_a3["total_rej_cap"]]
rejected_budget = [t_a1["total_rej_budget"], t_a3["total_rej_budget"]]
x_cat = np.arange(len(categories))
w = 0.5
p1 = ax.bar(x_cat, admitted, w, label="Accepted & Mutated", color="#48bb78", edgecolor="black")
p2 = ax.bar(x_cat, rejected_cap, w, bottom=admitted, label="Rejected by Inflow Cap", color="#ecc94b", edgecolor="black")
p3 = ax.bar(x_cat, rejected_budget, w, bottom=np.array(admitted)+np.array(rejected_cap), label="Rejected by Budget", color="#f56565", edgecolor="black")
ax.set_title("Candidate Admission Totals (V3)", fontsize=12, fontweight="bold")
ax.set_ylabel("Candidate Count")
ax.set_xticks(x_cat)
ax.set_xticklabels(categories)
ax.legend(fontsize=8, loc="upper left")

# (6) Quality Evolution (V1 -> V2 -> V3)
ax = axs[1, 2]
gens = ["V1 (Decoupled)", "V2 (Grace-Aware)", "V3 (Dynamic Inflow)"]
psnr_a1_trend = [records["A1"]["eval_30k"]["PSNR"], records["A1_G"]["eval_30k"]["PSNR"], records["A1_V3"]["eval_30k"]["PSNR"]]
psnr_a3_trend = [records["A3"]["eval_30k"]["PSNR"], records["A3_G"]["eval_30k"]["PSNR"], records["A3_V3"]["eval_30k"]["PSNR"]]
ssim_a1_trend = [records["A1"]["eval_30k"]["SSIM"] * 30, records["A1_G"]["eval_30k"]["SSIM"] * 30, records["A1_V3"]["eval_30k"]["SSIM"] * 30]

ax.plot(gens, psnr_a1_trend, marker="o", linewidth=2.5, color="#319795", label="A1 Evolution PSNR (dB)")
ax.plot(gens, psnr_a3_trend, marker="s", linewidth=2.5, color="#805ad5", label="A3 Evolution PSNR (dB)")
for i, txt in enumerate(psnr_a1_trend):
    ax.annotate(f"{txt:.2f} dB", (gens[i], txt + 0.3), ha="center", fontsize=9, fontweight="bold", color="#234e52")
for i, txt in enumerate(psnr_a3_trend):
    ax.annotate(f"{txt:.2f} dB", (gens[i], txt - 0.5), ha="center", fontsize=9, fontweight="bold", color="#44337a")

ax.set_title("Longitudinal Quality Trajectory (V1 -> V2 -> V3)", fontsize=12, fontweight="bold")
ax.set_ylabel("PSNR (dB)")
ax.set_ylim(17, 24)
ax.legend(fontsize=8.5, loc="lower right")

plt.tight_layout()
plot_path = repo_root / "experiments/results/v3_benchmark_comparison.png"
plt.savefig(plot_path, dpi=200)
plt.close()
print(f"Saved comparison plot to: {plot_path}")

# Print summary metrics for reporting
print("\n=== SUMMARY METRICS FOR REPORT ===")
print(f"A1_V3: Total requested={t_a1['total_requested']:,}, accepted={t_a1['total_accepted']:,}, rej_cap={t_a1['total_rej_cap']:,}, rej_budget={t_a1['total_rej_budget']:,}")
print(f"A1_V3: Total clones={t_a1['total_clones']:,}, splits={t_a1['total_splits']:,}, hyg_prunes={t_a1['total_hyg_prunes']:,}, age_prot={t_a1['total_age_prot']:,}, reset_prot={t_a1['total_reset_prot']:,}")
print(f"A3_V3: Total requested={t_a3['total_requested']:,}, accepted={t_a3['total_accepted']:,}, rej_cap={t_a3['total_rej_cap']:,}, rej_budget={t_a3['total_rej_budget']:,}")
print(f"A3_V3: Total clones={t_a3['total_clones']:,}, splits={t_a3['total_splits']:,}, hyg_prunes={t_a3['total_hyg_prunes']:,}, age_prot={t_a3['total_age_prot']:,}, reset_prot={t_a3['total_reset_prot']:,}")
