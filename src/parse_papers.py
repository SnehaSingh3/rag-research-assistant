"""
src/parse_papers.py
Converts every PDF in data/papers/ into markdown text in data/parsed/
and writes data/parse_report.csv so you can spot badly parsed papers.

"""

import csv
from pathlib import Path

import pymupdf
import pymupdf4llm

PAPERS_DIR = Path("data/papers")
PARSED_DIR = Path("data/parsed")
REPORT_CSV = Path("data/parse_report.csv")

MIN_CHARS_PER_PAGE = 1500  # below this, a paper is probably badly parsed


def parse_one(pdf_path: Path) -> dict:
    """Parse a single PDF and return stats about it."""
    with pymupdf.open(pdf_path) as doc:
        n_pages = len(doc)

    markdown = pymupdf4llm.to_markdown(str(pdf_path), show_progress=False)

    out_path = PARSED_DIR / f"{pdf_path.stem}.md"
    out_path.write_text(markdown, encoding="utf-8")

    n_chars = len(markdown)
    chars_per_page = n_chars // max(n_pages, 1)
    return {
        "file_name": pdf_path.name,
        "pages": n_pages,
        "characters": n_chars,
        "chars_per_page": chars_per_page,
        "flag": "CHECK" if chars_per_page < MIN_CHARS_PER_PAGE else "ok",
    }


def main():
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs\n")

    rows = []
    for i, pdf_path in enumerate(pdfs, start=1):
        try:
            stats = parse_one(pdf_path)
            print(f"[{i}/{len(pdfs)}] {stats['flag']:5} {pdf_path.name} "
                  f"({stats['pages']} pages, {stats['characters']:,} chars)")
            rows.append(stats)
        except Exception as e:
            print(f"[{i}/{len(pdfs)}] FAIL  {pdf_path.name}: {e}")
            rows.append({
                "file_name": pdf_path.name, "pages": 0, "characters": 0,
                "chars_per_page": 0, "flag": "FAILED",
            })

    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file_name", "pages", "characters", "chars_per_page", "flag"]
        )
        writer.writeheader()
        writer.writerows(rows)

    flagged = [r for r in rows if r["flag"] != "ok"]
    print("\n--- DONE ---")
    print(f"Parsed OK: {len(rows) - len(flagged)}")
    print(f"Need a look: {len(flagged)}")
    for r in flagged:
        print(f"   {r['flag']}: {r['file_name']}")


if __name__ == "__main__":
    main()