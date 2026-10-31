"""
src/evaluate_rerank.py
Same evaluation as evaluate_retrieval.py, but with a reranking step:
  1) vector search fetches a pool of candidates (default 20)
  2) a cross-encoder reads (question, chunk) pairs and re-orders the pool
  3) the top 10 after reranking are scored

    python src/evaluate_rerank.py --label ctx_rerank_minilm
    python src/evaluate_rerank.py --label ctx_rerank_bge --reranker BAAI/bge-reranker-base
"""

import argparse
import csv
import json
import time

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

# reuse the scoring helpers and settings from the first evaluation script
from evaluate_retrieval import (
    DB_DIR,
    DEFAULT_MODEL,
    DETAILS_DIR,
    KS,
    LOG_PATH,
    QUERY_PREFIX,
    QUESTIONS_PATH,
    first_rank,
    summarize,
)

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def chunk_for_reranker(meta, doc):
    """The reranker sees the paper title and section too, like the contextual embeddings."""
    return f"{meta['title']}\nSection: {meta['section']}\n{doc}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="rerank")
    parser.add_argument("--collection", default="chunks_300_50_ctx")
    parser.add_argument("--reranker", default=DEFAULT_RERANKER)
    parser.add_argument("--pool", type=int, default=20, help="candidates fetched before reranking")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(args.collection)

    embedder = SentenceTransformer(args.model)
    reranker = CrossEncoder(args.reranker, max_length=512)

    vectors = embedder.encode(
        [QUERY_PREFIX + q["question"] for q in questions],
        normalize_embeddings=True,
        batch_size=32,
    )
    pool = collection.query(
        query_embeddings=vectors.tolist(),
        n_results=args.pool,
        include=["metadatas", "documents"],
    )

    print(f"\nExperiment: {args.label}")
    print(f"Collection: {args.collection} | pool: {args.pool} | reranker: {args.reranker}")
    print("Reranking (this can take a few minutes on CPU)...\n")

    start = time.time()
    paper_ranks, strict_ranks, rows = [], [], []

    for q, metas, docs in zip(questions, pool["metadatas"], pool["documents"]):
        pairs = [[q["question"], chunk_for_reranker(m, d)] for m, d in zip(metas, docs)]
        scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
        order = sorted(range(len(pairs)), key=lambda i: -scores[i])[: max(KS)]
        top_metas = [metas[i] for i in order]
        top_docs = [docs[i] for i in order]

        r_paper = first_rank(top_metas, top_docs, q["expected"])
        r_strict = first_rank(top_metas, top_docs, q["expected"], q.get("must_contain"))
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

    print(f"\nReranking took {time.time() - start:.0f}s")

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

    misses = [(q, rp) for q, rp, rs in zip(questions, paper_ranks, strict_ranks) if rs is None]
    if misses:
        print("\nNot found in top 10 (strict):")
        for q, rp in misses:
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