"""
src/chunk_papers.py
Cleans the parsed papers in data/parsed/ and splits them into overlapping chunks.

    python src/chunk_papers.py                      # defaults: 300 words, 50 overlap
    python src/chunk_papers.py --size 500 --overlap 100
"""

import argparse
import csv
import json
import re
from pathlib import Path

PARSED_DIR = Path("data/parsed")
METADATA_CSV = Path("data/papers_metadata.csv")
CHUNKS_DIR = Path("data/chunks")

HEADING = re.compile(r"^(#{1,4})\s+(.*\S)\s*$")
REF_HEADING = re.compile(
    r"^\s*(#{1,6}\s*)?(\*\*|__)?\s*(\d+\.?\s*)?(references|bibliography)\s*(\*\*|__)?\s*$",
    re.IGNORECASE,
)
PICTURE_MARKER = re.compile(r"\*{0,2}=+>\s*picture.*?<=+\*{0,2}", re.IGNORECASE)


# ---------- cleaning ----------

def strip_references(text: str):
    """Cut the text at the last 'References' heading in the second part of the paper."""
    lines = text.split("\n")
    candidates = [
        i for i, line in enumerate(lines)
        if REF_HEADING.match(line) and i > 0.3 * len(lines)
    ]
    if not candidates:
        return text, False
    cut = candidates[-1]
    return "\n".join(lines[:cut]), True


def clean_text(text: str) -> str:
    text = PICTURE_MARKER.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------- sectioning ----------

def split_sections(text: str):
    """Split markdown into (section_title, body) pairs using headings."""
    sections, title, buffer = [], "Front matter", []
    for line in text.split("\n"):
        m = HEADING.match(line)
        if m:
            if "".join(buffer).strip():
                sections.append((title, "\n".join(buffer)))
            title = m.group(2).replace("*", "").strip()
            buffer = []
        else:
            buffer.append(line)
    if "".join(buffer).strip():
        sections.append((title, "\n".join(buffer)))
    return sections


# ---------- chunking ----------

def chunk_section(body: str, size: int, overlap: int):
    """Pack paragraphs into chunks of about `size` words, with `overlap` words carried over."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks, current, cur_words, has_new = [], [], 0, False

    for para in paragraphs:
        words = para.split()

        # a single huge paragraph: slide a window over its words
        if len(words) > size:
            if has_new:
                chunks.append("\n\n".join(current))
            current, cur_words, has_new = [], 0, False
            step = max(size - overlap, 1)
            for start in range(0, len(words), step):
                chunks.append(" ".join(words[start:start + size]))
                if start + size >= len(words):
                    break
            continue

        # current chunk is full: save it and start the next one with some overlap
        if cur_words + len(words) > size and has_new:
            chunks.append("\n\n".join(current))
            tail = " ".join(" ".join(current).split()[-overlap:]) if overlap else ""
            current = [tail] if tail else []
            cur_words = len(tail.split())
            has_new = False

        current.append(para)
        cur_words += len(words)
        has_new = True

    if has_new:
        chunks.append("\n\n".join(current))
    return chunks


# ---------- main ----------

def load_metadata():
    meta = {}
    if METADATA_CSV.exists():
        with open(METADATA_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                meta[Path(row["file_name"]).stem] = row
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=300, help="chunk size in words")
    parser.add_argument("--overlap", type=int, default=50, help="overlap in words")
    parser.add_argument("--min-words", type=int, default=30, help="drop chunks shorter than this")
    args = parser.parse_args()

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHUNKS_DIR / f"chunks_{args.size}_{args.overlap}.jsonl"
    meta = load_metadata()

    files = sorted(PARSED_DIR.glob("*.md"))
    print(f"Chunking {len(files)} papers (size={args.size}, overlap={args.overlap})\n")

    all_chunks, no_refs_found = [], []

    with open(out_path, "w", encoding="utf-8") as out:
        for md_file in files:
            stem = md_file.stem
            info = meta.get(stem, {})
            text = md_file.read_text(encoding="utf-8")

            text, stripped = strip_references(text)
            if not stripped:
                no_refs_found.append(md_file.name)
            text = clean_text(text)

            n = 0
            for section, body in split_sections(text):
                for chunk in chunk_section(body, args.size, args.overlap):
                    n_words = len(chunk.split())
                    if n_words < args.min_words:
                        continue
                    record = {
                        "chunk_id": f"{stem}::{n}",
                        "paper": stem,
                        "title": info.get("title", stem),
                        "year": info.get("year", ""),
                        "section": section,
                        "text": chunk,
                        "n_words": n_words,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    all_chunks.append(n_words)
                    n += 1
            print(f"  {md_file.name}: {n} chunks")

    total = len(all_chunks)
    print("\n--- DONE ---")
    print(f"Total chunks:        {total}")
    if total:
        print(f"Avg words per chunk: {sum(all_chunks) // total}")
        print(f"Min / max words:     {min(all_chunks)} / {max(all_chunks)}")
        print(f"Avg chunks per paper: {total // max(len(files), 1)}")
    print(f"References removed in {len(files) - len(no_refs_found)}/{len(files)} papers")
    if no_refs_found:
        print("No References heading found in:")
        for name in no_refs_found:
            print(f"   {name}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()