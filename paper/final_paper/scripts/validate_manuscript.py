from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from pypdf import PdfReader


PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parents[1]


def main():
    main_tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    section_text = "\n".join(p.read_text(encoding="utf-8") for p in sorted((PAPER / "sections").glob("*.tex")))
    all_tex = main_tex + "\n" + section_text
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = set()
    for group in re.findall(r"\\cite\w*\{([^}]+)\}", all_tex):
        cited.update(k.strip() for k in group.split(","))
    missing_citations = sorted(cited - bib_keys)
    missing_figures = []
    for fig in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", all_tex):
        if not (PAPER / fig).exists():
            missing_figures.append(fig)

    pdf = PdfReader(PAPER / "main.pdf")
    page_text = [p.extract_text() or "" for p in pdf.pages]
    reference_page = next((i + 1 for i, t in enumerate(page_text) if "References" in t), None)
    llm_page = next((i + 1 for i, t in enumerate(page_text) if "LLM Usage Statement" in t), None)
    log = (PAPER / "main.log").read_text(encoding="utf-8", errors="replace")
    with (PAPER / "quantitative_claims.csv").open(encoding="utf-8") as f:
        claims = list(csv.DictReader(f))

    checks = {
        "pdf_exists": (PAPER / "main.pdf").exists(),
        "total_pages": len(pdf.pages),
        "reference_start_page": reference_page,
        "content_page_count": (reference_page - 1) if reference_page else None,
        "content_within_cvpr_eight_page_limit": bool(reference_page and reference_page - 1 <= 8),
        "llm_statement_present": "LLM Usage Statement" in main_tex and llm_page is not None,
        "review_mode": "\\usepackage[review]{cvpr}" in main_tex,
        "anonymous_author": "Anonymous CVPR submission" in main_tex,
        "identifying_repository_url_absent": "github.com" not in all_tex.lower(),
        "acknowledgments_absent": "acknowledg" not in all_tex.lower(),
        "placeholder_tokens_absent": not re.search(r"\b(TODO|TBD|FIXME)\b", all_tex),
        "missing_citation_keys": missing_citations,
        "missing_figure_files": missing_figures,
        "undefined_citation_warning": bool(re.search(r"Citation.*undefined|There were undefined citations", log, re.I)),
        "overfull_boxes": bool(re.search(r"Overfull \\hbox|Overfull \\vbox", log)),
        "bibliography_entries": len(bib_keys),
        "cited_entries": len(cited),
        "quantitative_claims": len(claims),
        "verified_claims": sum(c["status"] == "VERIFIED" for c in claims),
        "unresolved_claims": sum(c["status"] != "VERIFIED" for c in claims),
        "main_figures": len(re.findall(r"\\begin\{figure\*?\}", all_tex)),
        "main_tables": len(re.findall(r"\\begin\{table\*?\}", all_tex)),
    }
    hard_pass = all([
        checks["pdf_exists"],
        checks["content_within_cvpr_eight_page_limit"],
        checks["llm_statement_present"],
        checks["review_mode"],
        checks["anonymous_author"],
        checks["identifying_repository_url_absent"],
        checks["acknowledgments_absent"],
        checks["placeholder_tokens_absent"],
        not missing_citations,
        not missing_figures,
        not checks["undefined_citation_warning"],
        not checks["overfull_boxes"],
        checks["unresolved_claims"] == 0,
    ])
    checks["overall_status"] = "PASS" if hard_pass else "FAIL"
    (PAPER / "MANUSCRIPT_VALIDATION.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    raise SystemExit(0 if hard_pass else 1)


if __name__ == "__main__":
    main()
