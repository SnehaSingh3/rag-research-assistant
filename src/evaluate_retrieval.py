"""
src/evaluate_retrieval.py
Scores retrieval quality on eval/questions.json using Hit@k and MRR.

    python src/evaluate_retrieval.py --label baseline
    python src/evaluate_retrieval.py --label chunks_500 --collection chunks_500_100
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
LOG_PATH = Path("eval/results_log.csv")
DETAILS_DIR = Path("eval/results")
KS = (1, 3, 5, 10)

sys.stdout.reconfigure(encoding="utf-8")


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def first_hit_rank(titles, expected):
    """Rank (1-based) of the first retrieved chunk whose paper title matches any expected keyword."""
    keywords = [norm(e) for e in expected]
    for rank, title in enumerate(titles, start=1):
        t = norm(title)
        if any(k in t for k in keywords):
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
    results = collection.query(query_embeddings=vectors.tolist(), n_results=max(KS))

    ranks, rows = [], []
    print(f"\nExperiment: {args.label}  (collection: {args.collection})\n")
    for q, metas in zip(questions, results["metadatas"]):
        titles = [m["title"] for m in metas]
        rank = first_hit_rank(titles, q["expected"])
        ranks.append(rank)
        rows.append({
            "id": q["id"], "difficulty": q["difficulty"], "rank": rank or "",
            "question": q["question"],
        })
        status = "ok  " if rank and rank <= 3 else ("low " if rank else "MISS")
        print(f"{status} rank={str(rank or '-'):>2}  Q{q['id']:<2} {q['question'][:80]}")

    overall = summarize(ranks)
    print("\n--- SUMMARY ---")
    print(f"Questions: {len(questions)}")
    for k in KS:
        print(f"Hit@{k:<2}: {overall[f'hit@{k}']:.2f}")
    print(f"MRR:    {overall['mrr']:.3f}")

    for level in sorted({q["difficulty"] for q in questions}):
        subset = [r for q, r in zip(questions, ranks) if q["difficulty"] == level]
        if subset:
            s = summarize(subset)
            print(f"[{level}] n={len(subset)}  Hit@5={s['hit@5']:.2f}  MRR={s['mrr']:.3f}")

    misses = [q for q, r in zip(questions, ranks) if r is None]
    if misses:
        print("\nNot found in top 10:")
        for q in misses:
            print(f"   Q{q['id']}: {q['question']}")

    # save per-question details
    DETAILS_DIR.mkdir(parents=True, exist_ok=True)
    details_path = DETAILS_DIR / f"{args.label}_details.csv"
    with open(details_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "difficulty", "rank", "question"])
        writer.writeheader()
        writer.writerows(rows)

    # append one summary row to the running log
    new_log = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_log:
            writer.writerow(["label", "collection", "n_questions",
                             "hit@1", "hit@3", "hit@5", "hit@10", "mrr"])
        writer.writerow([
            args.label, args.collection, len(questions),
            *(f"{overall[f'hit@{k}']:.3f}" for k in KS), f"{overall['mrr']:.3f}",
        ])

    print(f"\nDetails saved to {details_path}")
    print(f"Summary appended to {LOG_PATH}")


if __name__ == "__main__":
    main()