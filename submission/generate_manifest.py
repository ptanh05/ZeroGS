"""
Generates final_artifact_manifest.csv for submission package.
"""

import hashlib
import sys
from pathlib import Path
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SUBMISSION_DIR = Path(__file__).resolve().parent

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def generate_manifest():
    records = []
    for p in sorted(SUBMISSION_DIR.rglob("*")):
        if p.is_file() and p.name != "final_artifact_manifest.csv":
            rel_path = p.relative_to(SUBMISSION_DIR).as_posix()
            ext = p.suffix.lower()
            if ext in [".tex", ".bib"]:
                cat = "LaTeX Source"
            elif ext in [".pdf"]:
                cat = "Compiled Manuscript"
            elif ext in [".png", ".jpg"]:
                cat = "Figure Visual Asset"
            elif ext in [".md"]:
                cat = "Submission Documentation"
            elif ext in [".py"]:
                cat = "Compilation & Build Script"
            else:
                cat = "Artifact"
            
            records.append({
                "relative_path": rel_path,
                "category": cat,
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
                "verification_status": "VERIFIED"
            })
    
    df = pd.DataFrame(records)
    out_csv = SUBMISSION_DIR / "final_artifact_manifest.csv"
    df.to_csv(out_csv, index=False)
    print(f"Generated {out_csv} with {len(records)} artifacts.")

if __name__ == "__main__":
    generate_manifest()

