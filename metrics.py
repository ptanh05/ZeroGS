"""
ZeroGS: Quantitative Metrics Evaluation Pipeline (metrics.py).

Computes standard Novel View Synthesis evaluation metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
Exports metrics to JSON and outputs formatted console report.
"""

import os
import sys
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Any, List

import torch
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

from utils.image_utils import psnr
from utils.loss_utils import ssim

# Optional LPIPS support
try:
    import lpips
    lpips_fn = lpips.LPIPS(net="vgg")
    if torch.cuda.is_available():
        lpips_fn = lpips_fn.cuda()
    HAS_LPIPS = True
except (ImportError, Exception):
    lpips_fn = None
    HAS_LPIPS = False


def evaluate_folder(renders_dir: str, gts_dir: str, device: str = "cuda") -> Dict[str, float]:
    render_files = sorted([f for f in os.listdir(renders_dir) if f.lower().endswith((".png", ".jpg"))])
    gt_files = sorted([f for f in os.listdir(gts_dir) if f.lower().endswith((".png", ".jpg"))])

    assert len(render_files) > 0, f"No image files found in {renders_dir}"
    assert len(render_files) == len(gt_files), (
        f"Mismatch count: {len(render_files)} renders vs {len(gt_files)} ground truth images"
    )

    psnr_vals = []
    ssim_vals = []
    lpips_vals = []

    for r_name, g_name in zip(tqdm(render_files, desc="Calculating metrics"), gt_files):
        r_path = os.path.join(renders_dir, r_name)
        g_path = os.path.join(gts_dir, g_name)

        r_img = TF.to_tensor(Image.open(r_path).convert("RGB")).unsqueeze(0).to(device)
        g_img = TF.to_tensor(Image.open(g_path).convert("RGB")).unsqueeze(0).to(device)

        # PSNR & SSIM
        cur_psnr = psnr(r_img, g_img).mean().item()
        cur_ssim = ssim(r_img, g_img).item()
        psnr_vals.append(cur_psnr)
        ssim_vals.append(cur_ssim)

        # LPIPS
        if HAS_LPIPS and lpips_fn is not None:
            # lpips expects inputs normalized in [-1, 1]
            r_lp = r_img * 2.0 - 1.0
            g_lp = g_img * 2.0 - 1.0
            cur_lpips = lpips_fn(r_lp, g_lp).item()
            lpips_vals.append(cur_lpips)

    mean_psnr = float(sum(psnr_vals) / len(psnr_vals))
    mean_ssim = float(sum(ssim_vals) / len(ssim_vals))
    mean_lpips = float(sum(lpips_vals) / len(lpips_vals)) if lpips_vals else None

    return {
        "PSNR": mean_psnr,
        "SSIM": mean_ssim,
        "LPIPS": mean_lpips,
        "num_views": len(render_files),
    }


def find_render_gt_pairs(base_path: str) -> List[tuple]:
    pairs = []
    for root, dirs, _ in os.walk(base_path):
        if "renders" in dirs and "gt" in dirs:
            renders_path = os.path.join(root, "renders")
            gt_path = os.path.join(root, "gt")
            pairs.append((root, renders_path, gt_path))
    return pairs


def main():
    parser = ArgumentParser(description="ZeroGS Metrics Evaluator (PSNR, SSIM, LPIPS)")
    parser.add_argument(
        "-m", "--model_paths", nargs="+", required=True,
        help="Path(s) to model directory or folders containing renders and gt subdirectories."
    )
    parser.add_argument("-o", "--output_file", type=str, default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    overall_results = {}

    for model_path in args.model_paths:
        print(f"\n[ZeroGS Metrics] Inspecting: {model_path}")
        pairs = find_render_gt_pairs(model_path)
        if not pairs:
            # Check if model_path directly contains renders and gt
            if os.path.exists(os.path.join(model_path, "renders")) and os.path.exists(os.path.join(model_path, "gt")):
                pairs = [(model_path, os.path.join(model_path, "renders"), os.path.join(model_path, "gt"))]

        if not pairs:
            print(f"[-] No matching 'renders/' and 'gt/' folders found in {model_path}")
            continue

        for set_dir, r_dir, g_dir in pairs:
            print(f"[+] Evaluating: {set_dir}")
            res = evaluate_folder(r_dir, g_dir, device=device)
            overall_results[set_dir] = res

            # Print formatted table
            print(f"    Views evaluated: {res['num_views']}")
            print(f"    PSNR:  {res['PSNR']:.4f} dB")
            print(f"    SSIM:  {res['SSIM']:.4f}")
            if res['LPIPS'] is not None:
                print(f"    LPIPS: {res['LPIPS']:.4f}")
            else:
                print(f"    LPIPS: N/A (install lpips package to compute)")

            # Save json in set directory
            result_json_path = os.path.join(set_dir, "results.json")
            with open(result_json_path, "w") as f:
                json.dump(res, f, indent=2)
            print(f"    [+] Saved results to: {result_json_path}")

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(overall_results, f, indent=2)
        print(f"\n[+] Master results written to: {args.output_file}")


if __name__ == "__main__":
    main()
