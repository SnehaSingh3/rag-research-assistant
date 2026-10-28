"""
src/search_test.py
Quick test: ask a question, see which chunks the index returns.

    python src/search_test.py "How does Self-RAG decide when to retrieve?"
    python src/search_test.py "What is HyDE?" --k 3
"""

import argparse
import sys

import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = "chroma_db"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# BGE models work better when short queries carry this instruction prefix
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

sys.stdout.reconfigure(encoding="utf-8")  # avoids Windows console encoding errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5, help="how many chunks to return")
    parser.add_argument("--collection", default="chunks_300_50")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(args.collection)

    model = SentenceTransformer(args.model)
    query_vec = model.encode(QUERY_PREFIX + args.query, normalize_embeddings=True)

    results = collection.query(query_embeddings=[query_vec.tolist()], n_results=args.k)

    print(f"\nQuery: {args.query}\n")
    for rank, (doc, meta, dist) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0]), start=1
    ):
        similarity = 1 - dist  # cosine distance -> cosine similarity
        print(f"#{rank}  similarity={similarity:.3f}")
        print(f"    paper:   {meta['title']} ({meta['year']})")
        print(f"    section: {meta['section']}")
        preview = " ".join(doc.split())[:350]
        print(f"    text:    {preview}...\n")


if __name__ == "__main__":
    main()