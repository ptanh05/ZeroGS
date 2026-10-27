"""Measured CUDA stress test for admission control and prune-first.

Unlike ``stress_test_admission.py``, this script performs real tensor mutations,
reads CUDA allocator peaks, and records every failed or violating trial.
"""

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from zerogs.admission_controller import AdmissionAction, AdmissionController
from zerogs.cost_model import ByteCostModel
from zerogs.marginal_allocator import MarginalUtilityAllocator
from zerogs.memory_telemetry import capture_cuda_memory
from zerogs.mock_gaussian_model import MockGaussianModel


def cleanup_cuda(*objects: Any) -> None:
    del objects
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def candidate_masks(n_points: int, n_clone: int, n_split: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_clone + n_split > n_points:
        raise ValueError("Mutation candidates exceed the Gaussian population")
    clone_mask = torch.zeros(n_points, dtype=torch.bool, device=device)
    split_mask = torch.zeros(n_points, dtype=torch.bool, device=device)
    clone_mask[:n_clone] = True
    split_mask[n_clone:n_clone + n_split] = True
    return clone_mask, split_mask


def execute_selected_mutations(model, clone_mask, split_mask) -> None:
    if bool(clone_mask.any()):
        model.densify_and_clone(clone_mask, grad_threshold=0.0, scene_extent=1.0)
    if bool(split_mask.any()):
        model.densify_and_split(split_mask, grad_threshold=0.0, scene_extent=1.0)


def measured_grow_first_peak(
    n_base: int,
    n_clone: int,
    n_split: int,
    device: str,
) -> int:
    torch.manual_seed(7)
    model = MockGaussianModel(sh_degree=3, num_points=n_base, device=device)
    clone_mask, split_mask = candidate_masks(n_base, n_clone, n_split, device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    execute_selected_mutations(model, clone_mask, split_mask)
    torch.cuda.synchronize()
    peak = int(torch.cuda.max_memory_reserved())
    del clone_mask, split_mask, model
    cleanup_cuda()
    return peak


def run_prune_first_trial(
    trial_index: int,
    n_base: int,
    n_clone: int,
    n_split: int,
    grow_first_peak: int,
    cost_model: ByteCostModel,
    device: str,
    safety_headroom_mb: float,
) -> Dict[str, Any]:
    torch.manual_seed(7)
    model = MockGaussianModel(sh_degree=3, num_points=n_base, device=device)
    clone_mask, split_mask = candidate_masks(n_base, n_clone, n_split, device)
    demand = torch.linspace(0.0, 1.0, n_base, device=device)

    baseline = capture_cuda_memory()
    # Place the effective ceiling below the measured grow-first peak. The
    # desired margin scales with population but remains feasible to prune.
    target_margin = max(2 * 1024**2, int(0.04 * n_base * cost_model.persistent_bytes_per_gaussian))
    hard_budget_bytes = max(
        baseline.conservative_footprint_bytes + cost_model.cuda_padding_bytes,
        grow_first_peak - target_margin,
    )
    controller = AdmissionController(
        cost_model=cost_model,
        hard_vram_limit_mb=hard_budget_bytes / 1024**2,
        safety_headroom_mb=safety_headroom_mb,
        non_pytorch_allowance_mb=0.0,
    )
    prune_candidates = max(1, int(n_base * 0.30))
    initial_decision = controller.evaluate_admission(
        n_clone_requested=n_clone,
        n_split_requested=n_split,
        current_num_gaussians=n_base,
        prune_candidates_count=prune_candidates,
        current_allocated_bytes=baseline.allocated_bytes,
        current_reserved_bytes=baseline.reserved_bytes,
    )

    required = min(initial_decision.required_pre_prune, prune_candidates)
    prune_mask = torch.zeros(n_base, dtype=torch.bool, device=device)
    if required > 0:
        prune_mask[:required] = True
        model.prune_points(prune_mask)
        surviving = ~prune_mask
        clone_mask = clone_mask[surviving]
        split_mask = split_mask[surviving]
        demand = demand[surviving]
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    post_prune = capture_cuda_memory()
    final_decision = controller.evaluate_admission(
        n_clone_requested=initial_decision.n_clone,
        n_split_requested=initial_decision.n_split,
        current_num_gaussians=int(model.get_xyz.shape[0]),
        prune_candidates_count=0,
        current_allocated_bytes=post_prune.allocated_bytes,
        current_reserved_bytes=post_prune.reserved_bytes,
    )
    allocator = MarginalUtilityAllocator(cost_model)
    selected_clone, selected_split = allocator.joint_knapsack_allocate(
        demand,
        clone_mask,
        split_mask,
        final_decision.mutation_budget_bytes,
    )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    oom = False
    error = None
    try:
        execute_selected_mutations(model, selected_clone, selected_split)
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as exc:
        oom = True
        error = str(exc)
    final_memory = capture_cuda_memory()
    measured_peak = max(final_memory.peak_reserved_bytes, final_memory.peak_allocated_bytes)
    budget_violation = measured_peak > hard_budget_bytes

    result = {
        "trial": trial_index,
        "n_base": n_base,
        "n_clone_requested": n_clone,
        "n_split_requested": n_split,
        "initial_action": initial_decision.action.value,
        "required_pre_prune": required,
        "post_prune_action": final_decision.action.value,
        "n_clone_executed": int(selected_clone.sum().item()),
        "n_split_executed": int(selected_split.sum().item()),
        "hard_budget_mb": hard_budget_bytes / 1024**2,
        "safety_headroom_mb": safety_headroom_mb,
        "grow_first_peak_mb": grow_first_peak / 1024**2,
        "prune_first_peak_mb": measured_peak / 1024**2,
        "peak_reduction_mb": (grow_first_peak - measured_peak) / 1024**2,
        "oom": oom,
        "budget_violation": budget_violation,
        "error": error,
    }
    del model, clone_mask, split_mask, demand, prune_mask, selected_clone, selected_split
    cleanup_cuda()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Measured CUDA prune-first stress test")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--calibration", type=Path, default=Path("experiments/results/cost_calibration.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/real_cuda_stress_results.json"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--safety_headroom_mb", type=float, default=16.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; no synthetic fallback is permitted")
    cost_model = ByteCostModel.from_calibration_file(str(args.calibration))
    rng = random.Random(args.seed)
    results = []
    for trial in range(args.trials):
        n_base = rng.choice((60_000, 80_000, 100_000))
        n_clone = rng.choice((5_000, 10_000, 15_000))
        n_split = rng.choice((5_000, 10_000, 15_000))
        grow_peak = measured_grow_first_peak(n_base, n_clone, n_split, "cuda")
        result = run_prune_first_trial(
            trial, n_base, n_clone, n_split, grow_peak, cost_model, "cuda",
            args.safety_headroom_mb,
        )
        results.append(result)
        print(
            f"Trial {trial + 1:02d}/{args.trials}: {result['initial_action']} "
            f"pruned={result['required_pre_prune']} "
            f"peak={result['prune_first_peak_mb']:.1f}/{result['hard_budget_mb']:.1f}MB "
            f"violation={result['budget_violation']} oom={result['oom']}"
        )

    summary = {
        "evidence_type": "measured_cuda",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "trials": len(results),
        "safety_headroom_mb": args.safety_headroom_mb,
        "prune_first_trials": sum(
            row["initial_action"] in {
                AdmissionAction.PRUNE_FIRST.value,
                AdmissionAction.SHRINK_AND_PRUNE_FIRST.value,
            }
            for row in results
        ),
        "oom_trials": sum(row["oom"] for row in results),
        "budget_violation_trials": sum(row["budget_violation"] for row in results),
        "mean_peak_reduction_mb": sum(row["peak_reduction_mb"] for row in results) / len(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    if summary["oom_trials"] or summary["budget_violation_trials"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
