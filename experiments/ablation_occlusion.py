"""
Experiment 2: Occlusion-Aware Demand & Over-Pruning Ablation.
Synthetic logic probe for Module 4. It does not establish a PSNR recovery claim.

Quantitatively measures the fine structure survival rate under bottom-quantile pruning:
- Baseline (SPARE-GS): Uses naive demand D = H * V. Vulnerable to occlusion flicker;
  erroneously over-prunes thin foliage & branches.
- Proposed (ZeroGS): Uses occlusion-compensated demand D_tilde and protected structure mask,
  ensuring delicate structures are preserved.
"""

import os
import sys
import json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from zerogs.occlusion_engine import OcclusionAwareEngine, bottom_utility_mask


def run_occlusion_ablation(
    num_total_primitives: int = 50000,
    num_fine_structures: int = 5000,
    output_dir: str = "experiments/results",
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("ZeroGS Synthetic Occlusion-Aware Logic Probe (not measured PSNR evidence)")
    print(f"Total primitives: {num_total_primitives:,} | Delicate fine structures (foliage): {num_fine_structures:,}")
    print("=" * 80)

    # 1. Synthesize scene statistics
    # Primitives 0 .. num_fine_structures - 1 are thin foliage / branch structures:
    # High depth variation across cameras (visible from some angles, behind leaves in others),
    # low average transmittance (occluded boundary).
    # Primitives num_fine_structures .. end are smooth background / bulk geometry.

    scene_extent = 5.0
    engine = OcclusionAwareEngine(
        num_gaussians=num_total_primitives,
        gamma_occ=0.35,
        transmittance_tau=0.25,
        device=device,
    )

    # Simulate 5 camera viewpoints
    torch.manual_seed(42)
    for cam_id in range(5):
        vis_mask = torch.rand(num_total_primitives, device=device) > 0.4
        # Thin structures experience higher occlusion (only visible ~30% of the time)
        vis_mask[:num_fine_structures] = torch.rand(num_fine_structures, device=device) > 0.7
        vis_idx = torch.where(vis_mask)[0]

        depths = torch.full((num_total_primitives,), 2.5, device=device)
        # Foliage depth fluctuates across camera views
        depths[:num_fine_structures] = 2.0 + torch.randn(num_fine_structures, device=device) * 0.4
        view_depths = depths[vis_idx]

        T = torch.full((num_total_primitives,), 0.7, device=device)
        # Foliage experiences low transmittance at occlusion boundaries
        T[:num_fine_structures] = 0.15 + torch.rand(num_fine_structures, device=device) * 0.08
        view_T = T[vis_idx]

        engine.update_per_view_stats(vis_idx, view_depths, view_T)

    # Viewspace gradients (modest gradient on foliage)
    grads = torch.rand(num_total_primitives, device=device) * 0.005 + 0.001
    engine.grad_accum = grads

    # 2. Baseline SPARE-GS Demand: D_spare = H * V
    spare_demand = engine.grad_accum * torch.clamp(engine.accum_vis, min=1).float()

    # 3. ZeroGS Compensated Demand & Protected Mask
    zerogs_demand, protected_mask = engine.compute_compensated_demand(scene_extent=scene_extent)

    # 4. Measure Over-Pruning Rate at bottom 10% utility cutoff
    cutoff_quantile = 0.10

    # Under SPARE-GS:
    no_protection = torch.zeros_like(protected_mask)
    spare_pruned_mask = bottom_utility_mask(spare_demand, no_protection, cutoff_quantile)
    spare_foliage_pruned = spare_pruned_mask[:num_fine_structures].sum().item()
    spare_foliage_loss_rate = (spare_foliage_pruned / num_fine_structures) * 100

    # Under ZeroGS without protection:
    zerogs_unprotected_prune = bottom_utility_mask(zerogs_demand, no_protection, cutoff_quantile)
    zerogs_foliage_pruned_unprot = zerogs_unprotected_prune[:num_fine_structures].sum().item()

    # Under ZeroGS WITH Protected Structure Mask:
    zerogs_final_prune = bottom_utility_mask(zerogs_demand, protected_mask, cutoff_quantile)
    zerogs_foliage_pruned_final = zerogs_final_prune[:num_fine_structures].sum().item()
    zerogs_foliage_loss_rate = (zerogs_foliage_pruned_final / num_fine_structures) * 100

    protected_foliage_count = protected_mask[:num_fine_structures].sum().item()

    results = {
        "evidence_type": "simulation",
        "num_total_primitives": num_total_primitives,
        "num_fine_structures": num_fine_structures,
        "spare_foliage_pruned": spare_foliage_pruned,
        "spare_foliage_pruned_pct": round(spare_foliage_loss_rate, 2),
        "zerogs_foliage_protected_count": protected_foliage_count,
        "zerogs_foliage_pruned": zerogs_foliage_pruned_final,
        "zerogs_foliage_pruned_pct": round(zerogs_foliage_loss_rate, 2),
        "zerogs_unprotected_foliage_pruned": zerogs_foliage_pruned_unprot,
        "protection_effect_fewer_foliage_pruned": int(
            zerogs_foliage_pruned_unprot - zerogs_foliage_pruned_final
        ),
    }

    print(f"Visibility-only Foliage Pruned: {spare_foliage_pruned:,} / {num_fine_structures:,} ({spare_foliage_loss_rate:.1f}%)")
    print(f"ZeroGS Protected Foliage:  {protected_foliage_count:,} / {num_fine_structures:,} ({protected_foliage_count/num_fine_structures*100:.1f}%)")
    print(f"ZeroGS Foliage Pruned:     {zerogs_foliage_pruned_final:,} / {num_fine_structures:,} ({zerogs_foliage_loss_rate:.1f}%)")
    print(
        "Protection-only effect:     "
        f"{results['protection_effect_fewer_foliage_pruned']:,} fewer synthetic foliage points pruned"
    )

    # Save JSON summary
    summary_path = os.path.join(output_dir, "occlusion_ablation_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Ablation results saved to: {summary_path}")

    # Generate Chart
    plot_path = os.path.join(output_dir, "occlusion_ablation_comparison.png")
    plt.figure(figsize=(8, 5), dpi=300)
    methods = ["SPARE-GS (Naive Demand)", "ZeroGS (Occlusion-Aware + Protected)"]
    prune_rates = [spare_foliage_loss_rate, zerogs_foliage_loss_rate]
    colors = ["#d9534f", "#5cb85c"]

    bars = plt.bar(methods, prune_rates, color=colors, width=0.45)
    plt.ylabel("Delicate Foliage Over-Pruning Rate (%)", fontsize=12, fontweight="bold")
    plt.title("Synthetic Occlusion Ablation (Logic Validation Only)\nNo measured PSNR claim", fontsize=13, fontweight="bold")
    plt.ylim(0, max(prune_rates) * 1.3)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, rate in zip(bars, prune_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{rate:.1f}%", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"[+] Ablation chart saved to: {plot_path}")

    return results


if __name__ == "__main__":
    run_occlusion_ablation()
