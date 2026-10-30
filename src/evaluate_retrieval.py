"""
src/evaluate_retrieval.py
Scores retrieval quality on eval/questions.json.

Two levels of scoring:
  paper-level : a retrieved chunk comes from the right paper
  strict      : right paper AND the chunk contains one of the question's
                "must_contain" phrases (so it likely holds the actual answer)

Run from the project root:
    python src/evaluate_retrieval.py --label baseline_v3 --collection chunks_300_50
    python src/evaluate_retrieval.py --label ctx_v3 --collection chunks_300_50_ctx
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = "chroma_db"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

QUESTIONS_PATH = Path("eval/questions.json")
LOG_PATH = Path("eval/results_log_v2.csv")  # new log: has strict columns
DETAILS_DIR = Path("eval/results")
KS = (1, 3, 5, 10)

sys.stdout.reconfigure(encoding="utf-8")


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def squash(text: str) -> str:
    """Lowercase and drop all whitespace, so parser glitches like 'overallquality' don't matter."""
    return "".join(text.lower().split())


def first_rank(metas, docs, expected, must_contain=None):
    """1-based rank of the first chunk from an expected paper (and containing a phrase, if given)."""
    keywords = [norm(e) for e in expected]
    phrases = [squash(p) for p in (must_contain or [])]
    for rank, (meta, doc) in enumerate(zip(metas, docs), start=1):
        if not any(k in norm(meta["title"]) for k in keywords):
            continue
        if phrases and not any(p in squash(doc) for p in phrases):
            continue
        return rank
    return None


def summarize(ranks):
    n = len(ranks)
    out = {f"hit@{k}": sum(1 for r in ranks if r and r <= k) / n for k in KS}
    out["mrr"] = sum(1 / r for r in ranks if r) / n
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="baseline", help="name of this experiment")
    parser.add_argument("--collection", default="chunks_300_50")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(args.collection)
    model = SentenceTransformer(args.model)

    vectors = model.encode(
        [QUERY_PREFIX + q["question"] for q in questions],
        normalize_embeddings=True,
        batch_size=32,
    )
    results = collection.query(
        query_embeddings=vectors.tolist(),
        n_results=max(KS),
        include=["metadatas", "documents"],
    )

    paper_ranks, strict_ranks, rows = [], [], []
    print(f"\nExperiment: {args.label}  (collection: {args.collection})\n")
    for q, metas, docs in zip(questions, results["metadatas"], results["documents"]):
        r_paper = first_rank(metas, docs, q["expected"])
        r_strict = first_rank(metas, docs, q["expected"], q.get("must_contain"))
        paper_ranks.append(r_paper)
        strict_ranks.append(r_strict)
        rows.append({
            "id": q["id"], "difficulty": q["difficulty"],
            "paper_rank": r_paper or "", "strict_rank": r_strict or "",
            "question": q["question"],
        })
        status = "ok  " if r_strict and r_strict <= 3 else ("low " if r_strict else "MISS")
        print(f"{status} paper={str(r_paper or '-'):>2} strict={str(r_strict or '-'):>2}  "
              f"Q{q['id']:<2} {q['question'][:70]}")

    paper = summarize(paper_ranks)
    strict = summarize(strict_ranks)

    print("\n--- SUMMARY ---")
    print(f"Questions: {len(questions)}")
    print(f"{'':10}{'paper-level':>14}{'strict':>10}")
    for k in KS:
        print(f"Hit@{k:<5}{paper[f'hit@{k}']:>14.3f}{strict[f'hit@{k}']:>10.3f}")
    print(f"{'MRR':<10}{paper['mrr']:>14.3f}{strict['mrr']:>10.3f}")

    print("\nStrict scores by question type:")
    for level in sorted({q["difficulty"] for q in questions}):
        subset = [r for q, r in zip(questions, strict_ranks) if q["difficulty"] == level]
        s = summarize(subset)
        print(f"  [{level}] n={len(subset)}  Hit@1={s['hit@1']:.2f}  Hit@5={s['hit@5']:.2f}  MRR={s['mrr']:.3f}")

    strict_misses = [
        (q, rp) for q, rp, rs in zip(questions, paper_ranks, strict_ranks) if rs is None
    ]
    if strict_misses:
        print("\nNot found in top 10 (strict):")
        for q, rp in strict_misses:
            note = "right paper retrieved, answer chunk missing" if rp else "right paper not retrieved"
            print(f"   Q{q['id']}: {q['question'][:75]}  -> {note}")

    DETAILS_DIR.mkdir(parents=True, exist_ok=True)
    details_path = DETAILS_DIR / f"{args.label}_details.csv"
    with open(details_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "difficulty", "paper_rank", "strict_rank", "question"]
        )
        writer.writeheader()
        writer.writerows(rows)

    new_log = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_log:
            writer.writerow([
                "label", "collection", "n_questions",
                "paper_hit@1", "paper_hit@5", "paper_mrr",
                "strict_hit@1", "strict_hit@3", "strict_hit@5", "strict_hit@10", "strict_mrr",
            ])
        writer.writerow([
            args.label, args.collection, len(questions),
            f"{paper['hit@1']:.3f}", f"{paper['hit@5']:.3f}", f"{paper['mrr']:.3f}",
            *(f"{strict[f'hit@{k}']:.3f}" for k in KS), f"{strict['mrr']:.3f}",
        ])

    print(f"\nDetails saved to {details_path}")
    print(f"Summary appended to {LOG_PATH}")


if __name__ == "__main__":
    main()