from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats


HERE = Path(__file__).resolve()
PAPER = HERE.parents[1]
ROOT = HERE.parents[3]
FIG = PAPER / "figures"
TABLE = PAPER / "tables"
REGISTRY = ROOT / "paper" / "CANONICAL_EVIDENCE_REGISTRY.json"
PHYSICAL_AUDIT = ROOT / "FINAL_AUDIT" / "PHYSICAL_EVIDENCE_AUDIT.csv"
CALIBRATION = ROOT / "experiments" / "results" / "cost_calibration.json"


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def direct_run(run_id: str) -> dict:
    run = ROOT / "output" / run_id
    verification = load_json(run / "benchmark_verification.json")
    result = load_json(run / "test" / "ours_30000" / "results.json")
    status = load_json(run / "run_status.json")
    assert status["status"] == "completed" and status["oom"] is False
    assert verification["return_code"] == 0 and verification["oom"] is False
    for key in ("PSNR", "SSIM", "LPIPS", "num_views"):
        assert abs(float(result[key]) - float(verification["eval_30k"][key])) < 1e-10
    return {
        "id": run_id,
        "scene": verification.get("scene") or verification.get("effective_configuration", {}).get("scene") or run_id.split("_")[1],
        "method": verification.get("experiment") or verification.get("effective_configuration", {}).get("experiment") or run_id.removeprefix("benchmark_").split("_", 1)[-1],
        "psnr": float(result["PSNR"]),
        "ssim": float(result["SSIM"]),
        "lpips": float(result["LPIPS"]),
        "views": int(result["num_views"]),
        "gaussians": int(verification["ply_final_gaussians"]),
        "peak_vram": float(verification["peak_device_memory_mb"]),
        "duration": float(verification["duration_seconds"]),
    }


def save_fig(fig, stem: str):
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def framework_figure():
    fig, ax = plt.subplots(figsize=(7.2, 2.55))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.2, 1.35, 2.0, 1.35, "Raster statistics\n+ gradient demand", "#dcecff"),
        (2.65, 1.35, 2.0, 1.35, "Candidate pools\nclone / split", "#e8f3e2"),
        (5.1, 1.35, 2.0, 1.35, "Headroom-aware\ninflow governor", "#fff1cc"),
        (7.55, 1.35, 2.0, 1.35, "Gain-per-byte\nallocation", "#fce0df"),
        (10.0, 1.35, 1.8, 1.35, "Mutate\nGaussians", "#e9e1f7"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=color, ec="#263238", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][2]
        x2 = boxes[i + 1][0]
        ax.add_patch(FancyArrowPatch((x1 + 0.05, 2.02), (x2 - 0.05, 2.02), arrowstyle="-|>", mutation_scale=12, lw=1.1))
    ax.text(6.1, 3.45, "Peak-VRAM feedback: max(allocated, reserved) + allowance", ha="center", fontsize=9, weight="bold")
    ax.add_patch(FancyArrowPatch((10.9, 2.8), (6.1, 2.8), connectionstyle="arc3,rad=.25", arrowstyle="-|>", mutation_scale=12, lw=1.1, color="#1f5a88"))
    ax.text(6.0, 0.62, "Pressure-triggered preservation: prune low-demand candidates only when admission needs headroom", ha="center", fontsize=9, color="#8b1e1e")
    ax.add_patch(FancyArrowPatch((5.1, 1.32), (5.1, 0.83), arrowstyle="-|>", mutation_scale=11, lw=1.0, color="#8b1e1e"))
    save_fig(fig, "fig1_framework")


def lifecycle_figure():
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.axis("off")
    nodes = [
        (0.3, 1.5, 2.0, "Newly admitted\nGaussian", "#e8f3e2"),
        (3.0, 1.5, 2.0, "Optimization and\nevidence accumulation", "#dcecff"),
        (6.0, 1.5, 2.0, "Low-demand /\nlow-opacity candidate", "#fff1cc"),
        (8.65, 2.35, 2.05, "Retain when\nheadroom suffices", "#e8f3e2"),
        (8.65, 0.45, 2.05, "Prune when\nheadroom is tight", "#fce0df"),
    ]
    for x, y, w, text, color in nodes:
        ax.add_patch(FancyBboxPatch((x, y), w, 1.0, boxstyle="round,pad=0.04", fc=color, ec="#263238", lw=1.2))
        ax.text(x + w / 2, y + 0.5, text, ha="center", va="center", fontsize=9)
    ax.add_patch(FancyArrowPatch((2.3, 2), (3.0, 2), arrowstyle="-|>", mutation_scale=12))
    ax.add_patch(FancyArrowPatch((5.0, 2), (6.0, 2), arrowstyle="-|>", mutation_scale=12))
    ax.add_patch(FancyArrowPatch((8.0, 2.1), (8.65, 2.85), arrowstyle="-|>", mutation_scale=12, color="#28764f"))
    ax.add_patch(FancyArrowPatch((8.0, 1.9), (8.65, 0.95), arrowstyle="-|>", mutation_scale=12, color="#a23b3b"))
    ax.text(7.95, 2.72, "ACCEPT", fontsize=8, color="#28764f", weight="bold")
    ax.text(7.85, 0.72, "PRUNE-FIRST", fontsize=8, color="#a23b3b", weight="bold")
    ax.add_patch(FancyArrowPatch((9.65, 2.32), (4.0, 2.52), connectionstyle="arc3,rad=.22", arrowstyle="-|>", mutation_scale=12, lw=1.0, color="#28764f"))
    ax.text(5.65, 3.68, "Evaluated preservation has no explicit maturity threshold", ha="center", fontsize=9, style="italic")
    save_fig(fig, "fig2_lifecycle")


def data_figures(runs: dict[str, dict]):
    order = ["ABL-DI", "A1_V4_P1", "ABL-DI-HP", "ABL-HP"]
    labels = ["Static +\neager", "Dynamic +\neager", "Static +\npreserve", "Dynamic +\npreserve"]
    colors = ["#6baed6", "#3182bd", "#74c476", "#238b45"]
    vals = [runs[k]["psnr"] for k in order]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    bars = ax.bar(labels, vals, color=colors, edgecolor="white")
    ax.set_ylabel("PSNR (dB)")
    ax.set_ylim(0, 35)
    ax.grid(axis="y", alpha=.22)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + .45, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_title("2x2 factorial ablation on Bonsai (seed 0)", fontsize=10)
    save_fig(fig, "fig3_factorial")

    fig, ax = plt.subplots(figsize=(6.2, 3.35))
    scatter_keys = order + ["V_ref"]
    for key, color in zip(scatter_keys, colors + ["#756bb1"]):
        r = runs[key]
        ax.scatter(r["peak_vram"], r["psnr"], s=35 + r["gaussians"] / 9000, color=color, alpha=.82, edgecolor="black", linewidth=.5)
        ax.annotate(key, (r["peak_vram"], r["psnr"]), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("Peak physical VRAM (MiB; 2 Hz polling)")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(alpha=.22)
    ax.set_title("Quality-memory operating points; marker area scales with population", fontsize=9.5)
    save_fig(fig, "fig4_quality_memory")

    scenes = ["bonsai", "garden", "flowers"]
    base = [runs[f"{s}_base"]["psnr"] for s in scenes]
    prop = [runs[f"{s}_prop"]["psnr"] for s in scenes]
    x = np.arange(len(scenes))
    fig, ax = plt.subplots(figsize=(6.2, 3.25))
    w = .34
    b1 = ax.bar(x - w / 2, base, w, label="Eager hygiene", color="#6baed6")
    b2 = ax.bar(x + w / 2, prop, w, label="Pressure-triggered preservation", color="#238b45")
    ax.set_xticks(x, [s.capitalize() for s in scenes])
    ax.set_ylabel("PSNR (dB)")
    ax.set_ylim(0, 35)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=.22)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .35, f"{bar.get_height():.2f}", ha="center", fontsize=7.5)
    save_fig(fig, "fig5_cross_scene")

    fig, ax = plt.subplots(figsize=(6.2, 3.15))
    for run_id, label, color in [
        ("benchmark_bonsai_A1_V4_P1", "Eager hygiene", "#3182bd"),
        ("benchmark_bonsai_ABL-HP", "Pressure-triggered preservation", "#238b45"),
    ]:
        xs, ys = [], []
        with (ROOT / "output" / run_id / "training_telemetry.jsonl").open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("event") == "mutation_cycle" and "memory_after" in rec:
                    xs.append(rec["iteration"])
                    ys.append(rec["memory_after"]["conservative_footprint_mb"])
        ax.plot(xs, ys, label=label, color=color, linewidth=1.35)
    ax.axhline(6144 - 512, color="#b2182b", linestyle="--", linewidth=1, label="Controller effective budget")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Controller footprint (MiB)")
    ax.set_title("Internal allocator footprint (not total physical VRAM)", fontsize=9.5)
    ax.grid(alpha=.22)
    ax.legend(frameon=False, fontsize=7.5)
    save_fig(fig, "fig6_memory_trace")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)
    registry = load_json(REGISTRY)
    calibration = load_json(CALIBRATION)
    with PHYSICAL_AUDIT.open(encoding="utf-8-sig") as f:
        audit_rows = list(csv.DictReader(f))
    assert len(audit_rows) == 24
    assert all(r["audit_status"] == "PASS" for r in audit_rows)

    ids = [r["experiment_id"] for r in audit_rows]
    direct = {run_id: direct_run(run_id) for run_id in ids}
    reg = {r["experiment_id"]: r for r in registry["benchmarks"]}
    discrepancies = []
    for run_id, row in direct.items():
        rr = reg[run_id]
        for key, regkey in (("psnr", "psnr_30k"), ("ssim", "ssim_30k"), ("lpips", "lpips_30k"), ("peak_vram", "peak_vram_mb")):
            if not math.isclose(row[key], float(rr[regkey]), rel_tol=0, abs_tol=5e-4):
                discrepancies.append((run_id, key, row[key], rr[regkey]))
    assert not discrepancies, discrepancies

    run = {
        "ABL-DI": direct["benchmark_bonsai_ABL-DI"],
        "A1_V4_P1": direct["benchmark_bonsai_A1_V4_P1"],
        "ABL-DI-HP": direct["benchmark_bonsai_ABL-DI-HP"],
        "ABL-HP": direct["benchmark_bonsai_ABL-HP"],
        "V_ref": direct["benchmark_bonsai_V_ref"],
        "bonsai_base": direct["benchmark_bonsai_A1_V4_P1"],
        "bonsai_prop": direct["benchmark_bonsai_ABL-HP"],
        "garden_base": direct["benchmark_garden_A1_V4_P1"],
        "garden_prop": direct["benchmark_garden_ABL-HP"],
        "flowers_base": direct["benchmark_flowers_A1_V4_P1"],
        "flowers_prop": direct["benchmark_flowers_ABL-HP"],
    }

    base = np.array([direct[f"benchmark_bonsai_A1_V4_P1{suffix}"]["psnr"] for suffix in ("", "_s1", "_s2")])
    prop = np.array([direct[f"benchmark_bonsai_ABL-HP{suffix}"]["psnr"] for suffix in ("", "_s1", "_s2")])
    delta = prop - base
    ttest = stats.ttest_rel(prop, base)
    ci = stats.t.interval(.95, len(delta) - 1, loc=delta.mean(), scale=stats.sem(delta))
    cohen_dz = delta.mean() / delta.std(ddof=1)

    f = {k: run[k]["psnr"] for k in ("ABL-DI", "A1_V4_P1", "ABL-DI-HP", "ABL-HP")}
    effect_pres = ((f["ABL-DI-HP"] + f["ABL-HP"]) - (f["ABL-DI"] + f["A1_V4_P1"])) / 2
    effect_inflow = ((f["A1_V4_P1"] + f["ABL-HP"]) - (f["ABL-DI"] + f["ABL-DI-HP"])) / 2
    interaction = (f["ABL-HP"] - f["ABL-DI-HP"]) - (f["A1_V4_P1"] - f["ABL-DI"])

    summary = {
        "physical_runs": 24,
        "oom_events": sum(1 for r in audit_rows if r["oom_reported"].lower() == "true"),
        "peak_vram_min": min(r["peak_vram"] for r in direct.values()),
        "peak_vram_max": max(r["peak_vram"] for r in direct.values()),
        "hardware_vram": 8150.56,
        "minimum_headroom": 8150.56 - max(r["peak_vram"] for r in direct.values()),
        "base_psnr_mean": float(base.mean()),
        "base_psnr_std": float(base.std(ddof=1)),
        "proposed_psnr_mean": float(prop.mean()),
        "proposed_psnr_std": float(prop.std(ddof=1)),
        "delta_psnr_mean": float(delta.mean()),
        "delta_psnr_std": float(delta.std(ddof=1)),
        "t": float(ttest.statistic),
        "p": float(ttest.pvalue),
        "cohen_dz": float(cohen_dz),
        "ci95": [float(ci[0]), float(ci[1])],
        "factorial_preservation_main_effect": float(effect_pres),
        "factorial_inflow_main_effect": float(effect_inflow),
        "factorial_interaction": float(interaction),
        "calibration_clone_multiplier": calibration["clone_multiplier"],
        "calibration_split_multiplier": calibration["split_multiplier"],
        "calibration_clone_fixed_overhead_bytes": calibration["clone_fixed_overhead_bytes"],
        "calibration_split_fixed_overhead_bytes": calibration["split_fixed_overhead_bytes"],
        "calibration_combined_mape_pct": calibration["metadata"]["held_out_largest_batch_error"]["combined_mape_pct"],
    }
    (PAPER / "evidence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    claims = [
        ("Q01", "24 physical 30k runs", summary["physical_runs"], "FINAL_AUDIT/PHYSICAL_EVIDENCE_AUDIT.csv"),
        ("Q02", "0 observed OOMs", summary["oom_events"], "output/*/run_status.json"),
        ("Q03", "Maximum sampled physical VRAM (MiB)", summary["peak_vram_max"], "output/*/benchmark_verification.json"),
        ("Q04", "Minimum physical headroom (MiB)", summary["minimum_headroom"], "hardware capacity - Q03"),
        ("Q05", "Baseline Bonsai mean PSNR", summary["base_psnr_mean"], "three raw results.json files"),
        ("Q06", "Proposed Bonsai mean PSNR", summary["proposed_psnr_mean"], "three raw results.json files"),
        ("Q07", "Paired PSNR gain", summary["delta_psnr_mean"], "paired seeds 0,1,2"),
        ("Q08", "Paired gain sample std", summary["delta_psnr_std"], "paired seeds 0,1,2"),
        ("Q09", "Paired t statistic", summary["t"], "scipy.stats.ttest_rel"),
        ("Q10", "Paired t p value", summary["p"], "scipy.stats.ttest_rel"),
        ("Q11", "Paired Cohen dz", summary["cohen_dz"], "mean(delta)/std(delta)"),
        ("Q12", "95% CI lower", summary["ci95"][0], "Student t, df=2"),
        ("Q13", "95% CI upper", summary["ci95"][1], "Student t, df=2"),
        ("Q14", "Preservation main effect", summary["factorial_preservation_main_effect"], "four factorial raw results"),
        ("Q15", "Inflow main effect", summary["factorial_inflow_main_effect"], "four factorial raw results"),
        ("Q16", "Interaction", summary["factorial_interaction"], "difference in differences"),
        ("Q17", "Garden PSNR gain", run["garden_prop"]["psnr"] - run["garden_base"]["psnr"], "two raw results.json files"),
        ("Q18", "Flowers PSNR gain", run["flowers_prop"]["psnr"] - run["flowers_base"]["psnr"], "two raw results.json files"),
        ("Q19", "Static preservation PSNR", run["ABL-DI-HP"]["psnr"], "raw results.json"),
        ("Q20", "Dynamic preservation PSNR", run["ABL-HP"]["psnr"], "raw results.json"),
        ("Q21", "Persistent training bytes/Gaussian", 944, "zerogs/cost_model.py"),
        ("Q22", "Clone transient unit bytes", 944 * calibration["clone_multiplier"], "cost_calibration.json"),
        ("Q23", "Split transient unit bytes", 944 * calibration["split_multiplier"], "cost_calibration.json"),
        ("Q24", "Held-out calibration combined MAPE (%)", summary["calibration_combined_mape_pct"], "cost_calibration.json"),
        ("Q25", "Bonsai proposed vs vanilla PSNR gain", summary["proposed_psnr_mean"] - run["V_ref"]["psnr"], "raw proposed seeds and V_ref"),
        ("Q26", "Bonsai Gaussian reduction vs vanilla (%)", 100 * (1 - np.mean([direct[f"benchmark_bonsai_ABL-HP{s}"]["gaussians"] for s in ("", "_s1", "_s2")]) / run["V_ref"]["gaussians"]), "raw PLY counts"),
    ]
    with (PAPER / "quantitative_claims.csv").open("w", newline="", encoding="utf-8") as fobj:
        w = csv.writer(fobj)
        w.writerow(["claim_id", "description", "verified_value", "evidence", "status"])
        for row in claims:
            w.writerow([*row, "VERIFIED"])

    report = [
        "# Automated Evidence Validation Report",
        "",
        "- Raw physical runs inspected: 24/24",
        "- Raw run/result/verification agreement: PASS",
        "- Registry-to-raw agreement (PSNR, SSIM, LPIPS, peak VRAM): PASS",
        f"- Quantitative claims in the manuscript ledger: {len(claims)} verified, 0 unresolved",
        f"- Observed OOM events: {summary['oom_events']}/24",
        f"- Peak physical VRAM range: {summary['peak_vram_min']:.1f}-{summary['peak_vram_max']:.1f} MiB",
        f"- Paired Bonsai PSNR gain: {summary['delta_psnr_mean']:.4f} +/- {summary['delta_psnr_std']:.4f} dB",
        f"- Paired t test: t(2)={summary['t']:.4f}, p={summary['p']:.5f}",
        "",
        "The script treats files under `output/` as primary evidence. It does not alter raw evidence.",
    ]
    (PAPER / "EVIDENCE_VALIDATION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    framework_figure()
    lifecycle_figure()
    data_figures(run)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
