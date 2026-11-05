import json
import numpy as np
from pathlib import Path

def audit_run_telemetry(exp_name):
    path = Path(f"output/benchmark_bonsai_{exp_name}/training_telemetry.jsonl")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    mutation_cycles = [r for r in records if r.get("event") == "mutation_cycle"]
    assert len(mutation_cycles) == 144, f"Expected 144 cycles, got {len(mutation_cycles)}"
    
    # Dynamic cap
    caps = [r.get("computed_dynamic_cap") for r in mutation_cycles]
    mean_cap = np.mean(caps)
    max_cap = np.max(caps)
    min_cap = np.min(caps)
    
    # Candidate numbers
    requested = [r.get("requested_candidates", 0) for r in mutation_cycles]
    accepted = [r.get("accepted_candidates", 0) for r in mutation_cycles]
    rejected_cap = [r.get("rejected_by_cap", 0) for r in mutation_cycles]
    rejected_budget = [r.get("rejected_by_memory_budget", 0) for r in mutation_cycles]
    
    # Mutation executions
    clones = [r.get("n_clone_executed", 0) for r in mutation_cycles]
    splits = [r.get("n_split_executed", 0) for r in mutation_cycles]
    
    # Pruning executions
    hyg_prunes = [r.get("executed_hygiene_prune", 0) for r in mutation_cycles]
    pre_prunes = [r.get("executed_pre_prune", 0) for r in mutation_cycles]
    
    # Protections
    age_prot = [r.get("n_protected_by_age", 0) for r in mutation_cycles]
    reset_prot = [r.get("n_protected_by_cooldown", 0) for r in mutation_cycles]
    
    # Populations
    pops_before = [r.get("population_before") for r in mutation_cycles]
    pops_after = [r.get("population_after") for r in mutation_cycles]
    
    # Headroom
    headrooms = [r.get("available_headroom_mb", 0.0) for r in mutation_cycles]
    
    return {
        "exp": exp_name,
        "total_records": len(records),
        "mutation_cycles": len(mutation_cycles),
        "mean_dynamic_cap": float(mean_cap),
        "max_dynamic_cap": int(max_cap),
        "min_dynamic_cap": int(min_cap),
        "sum_requested": int(np.sum(requested)),
        "sum_accepted": int(np.sum(accepted)),
        "sum_rejected_by_cap": int(np.sum(rejected_cap)),
        "sum_rejected_by_budget": int(np.sum(rejected_budget)),
        "sum_clones": int(np.sum(clones)),
        "sum_splits": int(np.sum(splits)),
        "sum_newborn": int(np.sum(clones) + np.sum(splits)),
        "sum_hyg_prunes": int(np.sum(hyg_prunes)),
        "sum_pre_prunes": int(np.sum(pre_prunes)),
        "sum_age_protected": int(np.sum(age_prot)),
        "sum_reset_protected": int(np.sum(reset_prot)),
        "min_population": int(np.min(pops_after)),
        "max_population": int(np.max(pops_after)),
        "final_dens_population": int(pops_after[-1]),
        "mean_headroom_mb": float(np.mean(headrooms)),
        "min_headroom_mb": float(np.min(headrooms)),
        "max_headroom_mb": float(np.max(headrooms)),
        "cycles_at_max_cap": sum(1 for c in caps if c == 15000),
        "cycles_at_min_cap": sum(1 for c in caps if c == 5000),
    }

s_a1 = audit_run_telemetry("A1_V3")
s_a3 = audit_run_telemetry("A3_V3")

print("=== RECALCULATED TELEMETRY STATS ===")
print("A1_V3:", json.dumps(s_a1, indent=2))
print("A3_V3:", json.dumps(s_a3, indent=2))

