import json
import csv
from pathlib import Path

def parse_run(telemetry_path, name):
    records = []
    with open(telemetry_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    events = {}
    for r in records:
        ev = r.get("event", "unknown")
        events[ev] = events.get(ev, 0) + 1
    
    dens_records = [r for r in records if r.get("event") == "densification_admission_decision"]
    
    caps = [r.get("computed_dynamic_cap") for r in dens_records if r.get("computed_dynamic_cap") is not None]
    reqs = [r.get("requested_candidates", 0) for r in dens_records]
    accs = [r.get("accepted_candidates", 0) for r in dens_records]
    rej_caps = [r.get("rejected_by_cap", 0) for r in dens_records]
    rej_budgets = [r.get("rejected_by_memory_budget", 0) for r in dens_records]
    clones = [r.get("n_clone_executed", 0) for r in dens_records]
    splits = [r.get("n_split_executed", 0) for r in dens_records]
    pre_prunes = [r.get("executed_pre_prune", 0) for r in dens_records]
    hyg_prunes = [r.get("executed_hygiene_prune", 0) for r in dens_records]
    pop_befores = [r.get("population_before", 0) for r in dens_records]
    pop_afters = [r.get("population_after", 0) for r in dens_records]
    age_prot = [r.get("n_protected_by_age", 0) for r in dens_records]
    reset_prot = [r.get("n_protected_by_cooldown", 0) for r in dens_records]
    
    reclaims = [r for r in records if "reclaimed_mb" in r]
    
    return {
        "name": name,
        "event_counts": events,
        "total_dens_cycles": len(dens_records),
        "dynamic_cap_min": min(caps) if caps else 0,
        "dynamic_cap_max": max(caps) if caps else 0,
        "dynamic_cap_avg": sum(caps)/len(caps) if caps else 0,
        "total_requested": sum(reqs),
        "total_accepted": sum(accs),
        "total_rejected_cap": sum(rej_caps),
        "total_rejected_budget": sum(rej_budgets),
        "total_clones": sum(clones),
        "total_splits": sum(splits),
        "total_pre_prunes": sum(pre_prunes),
        "total_hyg_prunes": sum(hyg_prunes),
        "total_age_protected": sum(age_prot),
        "total_reset_protected": sum(reset_prot),
        "max_pop_during_dens": max(pop_afters) if pop_afters else 0,
        "final_pop_dens": pop_afters[-1] if pop_afters else 0,
        "reclaim_count": len(reclaims),
        "reclaims": [
            {
                "iteration": rc.get("iteration"),
                "reclaimed_mb": rc.get("reclaimed_mb"),
                "footprint_before_mb": rc.get("footprint_before_mb"),
                "footprint_after_mb": rc.get("footprint_after_mb")
            }
            for rc in reclaims
        ]
    }

a1_stats = parse_run("output/benchmark_bonsai_A1_V3/training_telemetry.jsonl", "A1_V3")
a3_stats = parse_run("output/benchmark_bonsai_A3_V3/training_telemetry.jsonl", "A3_V3")

with open("output/benchmark_bonsai_A1_V3/benchmark_verification.json", "r", encoding="utf-8") as f:
    a1_verif = json.load(f)

with open("output/benchmark_bonsai_A3_V3/benchmark_verification.json", "r", encoding="utf-8") as f:
    a3_verif = json.load(f)

print("=== A1_V3 TELEMETRY STATS ===")
for k, v in a1_stats.items():
    if k != "reclaims":
        print(f"  {k}: {v}")
print(f"  reclaims: {a1_stats['reclaims']}")

print("\n=== A3_V3 TELEMETRY STATS ===")
for k, v in a3_stats.items():
    if k != "reclaims":
        print(f"  {k}: {v}")
print(f"  reclaims: {a3_stats['reclaims']}")

