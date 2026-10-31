"""
src/rag_answer.py
End-to-end RAG pipeline over the research-paper index:
  1) vector search (top 20 chunks)
  2) cross-encoder reranking (keep the best 5)
  3) LLM answer using only those chunks, with [n] citation

    python src/rag_answer.py "How does Self-RAG decide when to retrieve?"
    python src/rag_answer.py "What is HyDE?" --show-chunks
    python src/rag_answer.py                      # interactive mode
"""

import argparse
import os
import sys

import chromadb
from dotenv import load_dotenv
from openai import APIError, OpenAI, RateLimitError
from sentence_transformers import CrossEncoder, SentenceTransformer

DB_DIR = "chroma_db"
COLLECTION = "chunks_300_50_ctx"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

NOT_FOUND = "I couldn't find this in the papers."

SYSTEM_PROMPT = f"""You are a research assistant that answers questions about a fixed collection of research papers on retrieval-augmented generation and retrieval for large language models.

Rules:
1. Answer ONLY using the numbered sources provided. Do not use outside knowledge.
2. Cite every claim with the source number in square brackets, like [1] or [2][3].
3. If the sources do not contain enough information to answer, reply exactly: "{NOT_FOUND}"
4. If the sources only partly answer the question, answer the part they cover and say what is missing.
5. Be concise: at most about 200 words unless the question asks for more detail.
6. Do not write a separate "Sources" list at the end; the application shows the sources itself."""


class RAGPipeline:
    def __init__(self, collection_name=COLLECTION, pool=20, top_k=5):
        load_dotenv()
        self.pool = pool
        self.top_k = top_k
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.reranker = CrossEncoder(RERANK_MODEL, max_length=512)
        self.collection = chromadb.PersistentClient(path=DB_DIR).get_collection(collection_name)
        self.llm = OpenAI(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ["LLM_API_KEY"],
        )
        self.llm_model = os.environ["LLM_MODEL"]

    # ---------- retrieval ----------
    def retrieve(self, question: str):
        """Vector search for a pool of candidates, then rerank and keep the best top_k."""
        vec = self.embedder.encode(QUERY_PREFIX + question, normalize_embeddings=True)
        res = self.collection.query(
            query_embeddings=[vec.tolist()],
            n_results=self.pool,
            include=["metadatas", "documents"],
        )
        metas, docs = res["metadatas"][0], res["documents"][0]

        pairs = [
            [question, f"{m['title']}\nSection: {m['section']}\n{d}"]
            for m, d in zip(metas, docs)
        ]
        scores = self.reranker.predict(pairs, batch_size=16, show_progress_bar=False)
        order = sorted(range(len(pairs)), key=lambda i: -scores[i])[: self.top_k]

        return [
            {
                "title": metas[i]["title"],
                "year": metas[i]["year"],
                "section": metas[i]["section"],
                "text": docs[i],
                "score": float(scores[i]),
            }
            for i in order
        ]

    # ---------- generation ----------
    @staticmethod
    def build_context(sources):
        blocks = []
        for n, s in enumerate(sources, start=1):
            blocks.append(
                f"[{n}] {s['title']} ({s['year']}), section: {s['section']}\n{s['text']}"
            )
        return "\n\n".join(blocks)

    def ask(self, question: str):
        sources = self.retrieve(question)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Sources:\n{self.build_context(sources)}\n\nQuestion: {question}",
            },
        ]
        try:
            response = self.llm.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.1,
                max_tokens=1500,  # reasoning models spend some of this thinking
            )
            answer = (response.choices[0].message.content or "").strip()
            if not answer:
                answer = "The model returned an empty answer. Please try again."
        except RateLimitError:
            answer = ("The language model is rate-limited right now (free tier). "
                      "Please wait a minute and try again.")
        except APIError as e:
            answer = f"The language model request failed: {e}"

        return {"answer": answer, "sources": sources}


def print_result(result, show_chunks=False):
    print("\n=== ANSWER ===")
    print(result["answer"])
    print("\n=== SOURCES ===")
    for n, s in enumerate(result["sources"], start=1):
        print(f"[{n}] {s['title']} ({s['year']}) - {s['section']}  (rerank score {s['score']:.2f})")
        if show_chunks:
            print("    " + " ".join(s["text"].split())[:500] + "...\n")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--show-chunks", action="store_true", help="print the retrieved passages")
    args = parser.parse_args()

    print("Loading models (about 20-30 seconds)...")
    rag = RAGPipeline()

    if args.question:
        print_result(rag.ask(args.question), args.show_chunks)
        return

    while True:  # interactive mode
        q = input("\nQuestion (or 'quit'): ").strip()
        if q.lower() in ("quit", "exit", "q", ""):
            break
        print_result(rag.ask(q), args.show_chunks)


if __name__ == "__main__":
    main()