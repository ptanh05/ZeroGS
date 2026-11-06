"""
Automated Verification Script for PV-SRA-3DGS Manuscript Claims & Tables.
Directly audits text and table assertions against disk artifacts.
"""

import json
import math
import sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

def run_verification():
    print("=" * 70)
    print("PV-SRA-3DGS AUTOMATED EVIDENCE & MANUSCRIPT CLAIM VERIFIER")
    print("=" * 70)

    ledger_path = REPO_ROOT / "experiments/results/MASTER_EXPERIMENT_LEDGER.csv"
    registry_path = REPO_ROOT / "paper/CANONICAL_EVIDENCE_REGISTRY.json"

    if not ledger_path.exists():
        print(f"[FAIL] Master ledger not found at {ledger_path}")
        sys.exit(1)
    if not registry_path.exists():
        print(f"[FAIL] Evidence registry not found at {registry_path}")
        sys.exit(1)

    df = pd.read_csv(ledger_path)
    with open(registry_path, "r", encoding="utf-8") as f:
        reg = json.load(f)

    errors = []
    checks_passed = 0

    # 1. Verify V_ref Baseline on Bonsai
    v_ref = df[(df["variant_method"] == "V_ref") & (df["scene"] == "bonsai") & (df["seed"] == 0)].iloc[0]
    if not math.isclose(v_ref["psnr_30k"], 29.5857, abs_tol=1e-3):
        errors.append(f"V_ref PSNR mismatch: {v_ref['psnr_30k']} vs 29.5857")
    else:
        checks_passed += 1

    if not math.isclose(v_ref["peak_vram_mb"], 7656.0, abs_tol=1.0):
        errors.append(f"V_ref peak VRAM mismatch: {v_ref['peak_vram_mb']} vs 7656.0")
    else:
        checks_passed += 1

    # 2. Verify Multi-Seed A1_V4_P1
    v4_runs = df[(df["variant_method"] == "A1_V4_P1") & (df["scene"] == "bonsai")]
    v4_psnr = v4_runs["psnr_30k"].values
    if len(v4_psnr) != 3:
        errors.append(f"A1_V4_P1 run count mismatch: expected 3, got {len(v4_psnr)}")
    else:
        mean_v4 = float(np.mean(v4_psnr))
        std_v4 = float(np.std(v4_psnr, ddof=1))
        if not math.isclose(mean_v4, 20.7455, abs_tol=1e-3):
            errors.append(f"A1_V4_P1 mean PSNR mismatch: {mean_v4:.4f} vs 20.7455")
        else:
            checks_passed += 1
        if not math.isclose(std_v4, 1.8691, abs_tol=1e-3):
            errors.append(f"A1_V4_P1 std PSNR mismatch: {std_v4:.4f} vs 1.8691")
        else:
            checks_passed += 1

    # 3. Verify Multi-Seed ABL-HP
    hp_runs = df[(df["variant_method"] == "ABL-HP") & (df["scene"] == "bonsai")]
    hp_psnr = hp_runs["psnr_30k"].values
    if len(hp_psnr) != 3:
        errors.append(f"ABL-HP run count mismatch: expected 3, got {len(hp_psnr)}")
    else:
        mean_hp = float(np.mean(hp_psnr))
        std_hp = float(np.std(hp_psnr, ddof=1))
        if not math.isclose(mean_hp, 31.8305, abs_tol=1e-3):
            errors.append(f"ABL-HP mean PSNR mismatch: {mean_hp:.4f} vs 31.8305")
        else:
            checks_passed += 1
        if not math.isclose(std_hp, 0.4789, abs_tol=1e-3):
            errors.append(f"ABL-HP std PSNR mismatch: {std_hp:.4f} vs 0.4789")
        else:
            checks_passed += 1

    # 4. Verify Paired Difference
    diffs = hp_psnr - v4_psnr
    mean_diff = float(np.mean(diffs))
    if not math.isclose(mean_diff, 11.0850, abs_tol=1e-3):
        errors.append(f"Paired gain mismatch: {mean_diff:.4f} vs 11.0850")
    else:
        checks_passed += 1

    # 5. Verify Cross-Scene Garden
    g_v4 = df[(df["variant_method"] == "A1_V4_P1") & (df["scene"] == "garden")].iloc[0]
    g_hp = df[(df["variant_method"] == "ABL-HP") & (df["scene"] == "garden")].iloc[0]
    if not math.isclose(g_v4["psnr_30k"], 18.6734, abs_tol=1e-3):
        errors.append(f"Garden V4 PSNR mismatch: {g_v4['psnr_30k']} vs 18.6734")
    else:
        checks_passed += 1
    if not math.isclose(g_hp["psnr_30k"], 24.9277, abs_tol=1e-3):
        errors.append(f"Garden HP PSNR mismatch: {g_hp['psnr_30k']} vs 24.9277")
    else:
        checks_passed += 1

    # 6. Verify Cross-Scene Flowers
    f_v4 = df[(df["variant_method"] == "A1_V4_P1") & (df["scene"] == "flowers")].iloc[0]
    f_hp = df[(df["variant_method"] == "ABL-HP") & (df["scene"] == "flowers")].iloc[0]
    if not math.isclose(f_v4["psnr_30k"], 17.3572, abs_tol=1e-3):
        errors.append(f"Flowers V4 PSNR mismatch: {f_v4['psnr_30k']} vs 17.3572")
    else:
        checks_passed += 1
    if not math.isclose(f_hp["psnr_30k"], 20.2994, abs_tol=1e-3):
        errors.append(f"Flowers HP PSNR mismatch: {f_hp['psnr_30k']} vs 20.2994")
    else:
        checks_passed += 1

    # 7. Verify ABL-DI-HP Factorial Run on Bonsai
    di_hp = df[(df["variant_method"] == "ABL-DI-HP") & (df["scene"] == "bonsai") & (df["seed"] == 0)].iloc[0]
    if not math.isclose(di_hp["psnr_30k"], 32.0227, abs_tol=1e-3):
        errors.append(f"ABL-DI-HP PSNR mismatch: {di_hp['psnr_30k']} vs 32.0227")
    else:
        checks_passed += 1
    if not math.isclose(di_hp["ssim_30k"], 0.9456, abs_tol=1e-3):
        errors.append(f"ABL-DI-HP SSIM mismatch: {di_hp['ssim_30k']} vs 0.9456")
    else:
        checks_passed += 1
    if int(di_hp["final_gaussians"]) != 653344:
        errors.append(f"ABL-DI-HP Gaussians mismatch: {di_hp['final_gaussians']} vs 653344")
    else:
        checks_passed += 1

    # 8. Check OOM Count Across Entire Ledger (24 runs)
    if len(df) != 24:
        errors.append(f"Expected 24 benchmark runs in ledger, found {len(df)}")
    else:
        checks_passed += 1
    ooms = df[df["oom"] == True]
    if len(ooms) > 0:
        errors.append(f"Found {len(ooms)} runs with OOM reported!")
    else:
        checks_passed += 1

    print(f"Total assertions verified: {checks_passed}")
    if errors:
        print(f"[FAIL] Encountered {len(errors)} discrepancies:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("[PASS] All manuscript claims match physical experimental artifacts perfectly!")
        return True

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
