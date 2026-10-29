"""
src/build_index.py
Embeds every chunk and stores it in a local ChromaDB collection (folder: chroma_db/).

    python src/build_index.py
    python src/build_index.py --chunks data/chunks/chunks_500_100.jsonl   # another chunk setting
"""

import argparse
import json
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = "chroma_db"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
BATCH = 256  # how many chunks we send to ChromaDB at a time


def load_chunks(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/chunks/chunks_300_50.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--collection", default=None,
                        help="collection name (default: taken from the chunk file name)")
    parser.add_argument("--with-context", action="store_true",
                        help="prepend paper title and section to each chunk before embedding")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    collection_name = args.collection or chunks_path.stem  # e.g. chunks_300_50
    if args.with_context and not args.collection:
        collection_name += "_ctx"  # e.g. chunks_300_50_ctx

    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    print(f"Loading embedding model: {args.model} (first run downloads it, ~130 MB)")
    model = SentenceTransformer(args.model)

    print("Embedding chunks (this takes a few minutes on CPU)...")
    start = time.time()
    if args.with_context:
        # the model "sees" which paper and section each chunk comes from
        texts = [f"{c['title']}\nSection: {c['section']}\n{c['text']}" for c in chunks]
    else:
        texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,  # needed for cosine similarity
        show_progress_bar=True,
    )
    print(f"Embedding done in {time.time() - start:.0f}s, vector size = {embeddings.shape[1]}")

    client = chromadb.PersistentClient(path=DB_DIR)
    try:  # start fresh so re-runs don't create duplicates
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )

    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=embeddings[i:i + BATCH].tolist(),
            metadatas=[
                {
                    "paper": c["paper"],
                    "title": c["title"],
                    "year": str(c["year"]),
                    "section": c["section"],
                    "n_words": c["n_words"],
                }
                for c in batch
            ],
        )

    print("\n--- DONE ---")
    print(f"Collection '{collection_name}' now holds {collection.count()} chunks")
    print(f"Saved in folder: {DB_DIR}/")


if __name__ == "__main__":
    main()