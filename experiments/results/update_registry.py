import json
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# 1. Update CSV
df = pd.read_csv(REPO_ROOT / "experiments/results/MASTER_EXPERIMENT_LEDGER.csv")
df.to_csv(REPO_ROOT / "paper/CANONICAL_EVIDENCE_REGISTRY.csv", index=False)
print("Updated paper/CANONICAL_EVIDENCE_REGISTRY.csv")

# 2. Update JSON
json_path = REPO_ROOT / "paper/CANONICAL_EVIDENCE_REGISTRY.json"
with open(json_path, "r", encoding="utf-8") as f:
    reg = json.load(f)

reg["metadata"]["total_physical_runs"] = len(df)

# Convert dataframe records to benchmarks list
benchmarks = []
for _, row in df.iterrows():
    benchmarks.append({k: str(v) if pd.notna(v) else "" for k, v in row.items()})

reg["benchmarks"] = benchmarks

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(reg, f, indent=2)

print("Updated paper/CANONICAL_EVIDENCE_REGISTRY.json with", len(benchmarks), "runs.")
