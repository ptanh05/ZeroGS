"""
Download and extract the NeRF Synthetic 'lego' scene for ZeroGS real-world testing.
"""

import os
import sys
import zipfile
import urllib.request
from tqdm import tqdm


DATA_URL = "https://huggingface.co/datasets/XayahHina/nerf_synthetic/resolve/main/nerf_synthetic.zip"
TARGET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
LEGO_DIR = os.path.join(TARGET_DIR, "lego")
ZIP_PATH = os.path.join(TARGET_DIR, "nerf_synthetic.zip")


def download_with_progress(url: str, output_path: str):
    print(f"[1/3] Downloading NeRF Synthetic dataset from:\n  {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        block_size = 1024 * 1024  # 1 MB blocks
        
        with open(output_path, "wb") as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                f.write(buffer)
                bar.update(len(buffer))


def extract_lego_scene(zip_path: str, extract_root: str):
    print(f"\n[2/3] Extracting only 'lego' scene files...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        lego_members = [m for m in zf.namelist() if m.startswith("lego/")]
        for m in tqdm(lego_members, desc="Extracting lego"):
            zf.extract(m, extract_root)
    print(f"Extraction complete: {len(lego_members)} files in {LEGO_DIR}")


def main():
    os.makedirs(TARGET_DIR, exist_ok=True)

    if os.path.exists(os.path.join(LEGO_DIR, "transforms_train.json")):
        print(f"[ZeroGS] 'lego' dataset is already present at:\n  {LEGO_DIR}")
        return

    # Step 1: Download
    if not os.path.exists(ZIP_PATH):
        download_with_progress(DATA_URL, ZIP_PATH)
    else:
        print(f"[ZeroGS] Found existing zip file: {ZIP_PATH}")

    # Step 2: Extract only lego
    extract_lego_scene(ZIP_PATH, TARGET_DIR)

    # Step 3: Cleanup zip archive to save space
    if os.path.exists(ZIP_PATH):
        print("[3/3] Removing zip archive to conserve disk space...")
        try:
            os.remove(ZIP_PATH)
        except Exception as e:
            print(f"Note: Could not remove zip file: {e}")

    print(f"\n[ZeroGS] Lego scene ready at: {LEGO_DIR}")
    print("Files verified:")
    print(" - transforms_train.json:", os.path.exists(os.path.join(LEGO_DIR, "transforms_train.json")))
    print(" - transforms_test.json:", os.path.exists(os.path.join(LEGO_DIR, "transforms_test.json")))


if __name__ == "__main__":
    main()
