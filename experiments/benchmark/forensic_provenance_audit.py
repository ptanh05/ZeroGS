import json
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from plyfile import PlyData
from PIL import Image

for exp in ["A1_V3", "A3_V3"]:
    base = Path(f"output/benchmark_bonsai_{exp}")
    print(f"\n==========================================")
    print(f"=== PROVENANCE AUDIT: {exp} ===")
    print(f"==========================================")
    
    # 1. run_manifest
    manifest_p = base / "run_manifest.json"
    if manifest_p.exists():
        with open(manifest_p, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            print("Manifest keys:", list(manifest.keys()))
            args = manifest.get("args", {})
            print(f"Dataset: {args.get('source_path')}")
            print(f"Images folder: {args.get('images')}")
            print(f"Resolution flag: {args.get('resolution')}")
            print(f"Seed: {args.get('seed')}")
            print(f"Iterations: {args.get('iterations')}")
            print(f"Pruning mode: {args.get('pruning_mode')}")
            print(f"Occlusion mode: {args.get('occlusion_physics_mode')}")
            print(f"Inflow mode: {args.get('inflow_mode')}")
            print(f"Inflow cap min: {args.get('inflow_cap_min')}")
            print(f"Inflow cap max: {args.get('inflow_cap_max')}")
            print(f"Target headroom: {args.get('target_headroom_mb')}")
            print(f"Tau age: {args.get('tau_age')}, Tau reset: {args.get('tau_reset')}, Kappa split: {args.get('kappa_split')}")
            print(f"Cost calibration: {args.get('cost_calibration')}")
    else:
        print("run_manifest.json MISSING!")
    
    # 2. run_status
    status_p = base / "run_status.json"
    if status_p.exists():
        with open(status_p, "r", encoding="utf-8") as f:
            print("Run status:", json.load(f))
    else:
        print("run_status.json MISSING!")
            
    # 3. PLY vertex count
    ply_30k = base / "point_cloud/iteration_30000/point_cloud.ply"
    if ply_30k.exists():
        plydata = PlyData.read(str(ply_30k))
        v_count_30k = len(plydata["vertex"])
        print(f"PLY 30K vertex count: {v_count_30k:,}")
    else:
        print("point_cloud.ply at 30K MISSING!")
        
    ply_7k = base / "point_cloud/iteration_7000/point_cloud.ply"
    if ply_7k.exists():
        plydata7 = PlyData.read(str(ply_7k))
        v_count_7k = len(plydata7["vertex"])
        print(f"PLY 7K vertex count: {v_count_7k:,}")
    else:
        print("point_cloud.ply at 7K MISSING!")
        
    # 4. Checkpoints
    for it in [7000, 30000]:
        ckpt = base / f"chkpnt{it}.pth"
        print(f"Checkpoint {it} exists: {ckpt.exists()}, size: {ckpt.stat().st_size if ckpt.exists() else 0:,} bytes")
        
    # 5. Rendered images count & dimensions
    for it in [7000, 30000]:
        renders_dir = base / f"test/ours_{it}/renders"
        gt_dir = base / f"test/ours_{it}/gt"
        res_json = base / f"test/ours_{it}/results.json"
        
        r_imgs = sorted(list(renders_dir.glob("*.png"))) if renders_dir.exists() else []
        g_imgs = sorted(list(gt_dir.glob("*.png"))) if gt_dir.exists() else []
        print(f"Eval {it}: renders={len(r_imgs)}, gt={len(g_imgs)}, results_json={res_json.exists()}")
        if r_imgs:
            with Image.open(r_imgs[0]) as im:
                print(f"  First render ({r_imgs[0].name}) resolution: {im.size} (WxH)")
            with Image.open(g_imgs[0]) as im:
                print(f"  First GT ({g_imgs[0].name}) resolution: {im.size} (WxH)")
        if res_json.exists():
            with open(res_json, "r", encoding="utf-8") as f:
                res = json.load(f)
                m = res.get("our_metrics", {})
                print(f"  Metrics: PSNR={m.get('PSNR')}, SSIM={m.get('SSIM')}, LPIPS={m.get('LPIPS')}")

