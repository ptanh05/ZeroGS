import os
import sys
import json
from pathlib import Path

def audit_datasets():
    print("=" * 60)
    print("1. DATASET AND SPLIT CONFIGURATION AUDIT")
    print("=" * 60)
    
    # Read dataset_readers.py to see LLFF hold rule
    try:
        with open("scene/dataset_readers.py", "r", encoding="utf-8") as f:
            code = f.read()
        llffhold_lines = [l.strip() for l in code.splitlines() if "llffhold" in l.lower() or "eval" in l.lower() and "i %" in l]
        print("LLFF Hold rule in code:")
        for l in llffhold_lines[:10]:
            print("  ", l)
    except Exception as e:
        print("Error reading dataset_readers.py:", e)
        
    for scene in ["bonsai", "garden", "flowers"]:
        p = Path("data") / scene
        print(f"\nScene: {scene} ({p})")
        if not p.exists():
            print("  DIRECTORY NOT FOUND")
            continue
        images_folders = sorted([d.name for d in p.iterdir() if d.is_dir() and "image" in d.name.lower()])
        for img_f in images_folders:
            files = [f for f in (p / img_f).iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".png", ".jpeg"]]
            print(f"  {img_f}: {len(files)} image files")
            
        sparse = p / "sparse" / "0"
        if sparse.exists():
            print(f"  sparse/0 files: {[f.name for f in sparse.iterdir()]}")
            
def audit_benchmarks():
    print("\n" + "=" * 60)
    print("2. PHYSICAL BENCHMARK RUN AUDIT (FROM RAW ARTIFACTS)")
    print("=" * 60)
    output_dir = Path("output")
    bench_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("benchmark_") and not "interrupted" in d.name])
    print(f"Found {len(bench_dirs)} non-interrupted benchmark directories.")
    
    records = []
    for d in bench_dirs:
        verif_file = d / "benchmark_verification.json"
        metrics_file = d / "metrics_summary.json"
        chkpnt = d / "chkpnt30000.pth"
        ply = d / "point_cloud" / "iteration_30000" / "point_cloud.ply"
        telemetry = d / "training_telemetry.jsonl"
        results_7k = d / "test" / "ours_7000" / "results.json"
        results_30k = d / "test" / "ours_30000" / "results.json"
        
        info = {
            "dir_name": d.name,
            "has_verif": verif_file.exists(),
            "has_metrics": metrics_file.exists(),
            "has_chkpnt": chkpnt.exists(),
            "has_ply": ply.exists(),
            "has_telemetry": telemetry.exists(),
            "has_results_7k": results_7k.exists(),
            "has_results_30k": results_30k.exists(),
            "chkpnt_size_bytes": chkpnt.stat().st_size if chkpnt.exists() else 0,
            "ply_size_bytes": ply.stat().st_size if ply.exists() else 0,
            "telemetry_lines": sum(1 for _ in open(telemetry, "rb")) if telemetry.exists() else 0,
        }
        
        # Load verification JSON if present
        if verif_file.exists():
            try:
                with open(verif_file, "r", encoding="utf-8") as f:
                    v = json.load(f)
                info["verif_experiment"] = v.get("experiment")
                info["verif_scene"] = v.get("scene")
                info["verif_seed"] = v.get("effective_configuration", {}).get("seed", 0)
                info["verif_iterations"] = v.get("effective_configuration", {}).get("iterations", 30000)
                info["verif_return_code"] = v.get("return_code")
                info["verif_oom"] = v.get("oom")
                info["verif_peak_vram_mb"] = v.get("peak_device_memory_mb")
                info["verif_ply_gaussians"] = v.get("ply_final_gaussians")
                info["verif_duration_s"] = v.get("duration_seconds")
                info["verif_eval_views"] = v.get("expected_eval_views")
                info["verif_psnr_30k"] = v.get("eval_30k", {}).get("PSNR")
                info["verif_ssim_30k"] = v.get("eval_30k", {}).get("SSIM")
                info["verif_lpips_30k"] = v.get("eval_30k", {}).get("LPIPS")
                info["verif_git_commit"] = v.get("git", {}).get("commit")
                info["cmd_flags"] = v.get("effective_configuration", {}).get("command_line_flags", [])
            except Exception as e:
                info["verif_error"] = str(e)
                
        # Also load results_30k directly
        if results_30k.exists():
            try:
                with open(results_30k, "r", encoding="utf-8") as f:
                    r30 = json.load(f)
                info["raw_psnr_30k"] = r30.get("PSNR")
                info["raw_ssim_30k"] = r30.get("SSIM")
                info["raw_lpips_30k"] = r30.get("LPIPS")
            except Exception as e:
                info["raw_results_error"] = str(e)
                
        # Check candidate accounting from telemetry
        if telemetry.exists():
            tot_req = 0
            tot_adm = 0
            tot_rej_cap = 0
            tot_rej_dyn = 0
            mut_cycles = 0
            with open(telemetry, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("event") == "mutation_cycle":
                            mut_cycles += 1
                            dec = entry.get("decision", {})
                            dyn = dec.get("dynamic_inflow", {})
                            n_clone = entry.get("n_clone_executed", 0)
                            n_split = entry.get("n_split_executed", 0)
                            tot_adm += (n_clone + n_split)
                            if dyn:
                                tot_req += dyn.get("n_requested", 0)
                                tot_rej_cap += dyn.get("rejected_by_cap", 0)
                                tot_rej_dyn += dyn.get("rejected_by_budget", 0)
                    except Exception:
                        pass
            info["mut_cycles"] = mut_cycles
            info["tot_req"] = tot_req
            info["tot_adm"] = tot_adm
            info["tot_rej_cap"] = tot_rej_cap
            info["tot_rej_dyn"] = tot_rej_dyn
            info["candidate_delta"] = tot_req - (tot_adm + tot_rej_cap + tot_rej_dyn) if tot_req > 0 else 0

        records.append(info)
        
    # Save audited raw table to JSON for further processing
    with open("experiments/results/raw_forensic_records.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Audited {len(records)} runs and saved raw forensic records.")
    
    # Print summary table
    print("\nSummary of Audited Physical Runs:")
    print(f"{'Directory':<30} | {'Method':<10} | {'Scene':<7} | {'Seed':<4} | {'PSNR':<8} | {'SSIM':<7} | {'LPIPS':<7} | {'Gaussians':<9} | {'Peak VRAM':<9} | {'RC':<3}")
    print("-" * 115)
    for r in records:
        method = str(r.get("verif_experiment") or "N/A")
        scene = str(r.get("verif_scene") or "N/A")
        seed = str(r.get("verif_seed", "N/A"))
        psnr = f"{r.get('raw_psnr_30k', 0.0):.4f}" if r.get('raw_psnr_30k') is not None else "N/A"
        ssim = f"{r.get('raw_ssim_30k', 0.0):.4f}" if r.get('raw_ssim_30k') is not None else "N/A"
        lpips = f"{r.get('raw_lpips_30k', 0.0):.4f}" if r.get('raw_lpips_30k') is not None else "N/A"
        gauss = str(r.get("verif_ply_gaussians") if r.get("verif_ply_gaussians") is not None else "N/A")
        vram = f"{r.get('verif_peak_vram_mb', 0.0):.1f}" if r.get('verif_peak_vram_mb') is not None else "N/A"
        rc = str(r.get("verif_return_code") if r.get("verif_return_code") is not None else "N/A")
        print(f"{r['dir_name']:<30} | {method:<10} | {scene:<7} | {seed:<4} | {psnr:<8} | {ssim:<7} | {lpips:<7} | {gauss:<9} | {vram:<9} | {rc:<3}")

if __name__ == "__main__":
    audit_datasets()
    audit_benchmarks()
