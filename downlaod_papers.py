"""
download_papers.py
Downloads the RAG paper set from arXiv by title and builds papers_metadata.csv.

Run from your project root:
    pip install arxiv
    python download_papers.py
"""

import csv
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

import arxiv

PAPERS_DIR = Path("data/papers")
METADATA_CSV = Path("data/papers_metadata.csv")
NOT_FOUND_TXT = Path("data/not_found.txt")
MATCH_THRESHOLD = 0.80  # how similar the arXiv title must be to ours

# (subtopic, title to search for)
PAPERS = [
    # A. Foundations
    ("foundations", "Attention Is All You Need"),
    ("foundations", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"),
    ("foundations", "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"),
    ("foundations", "Lost in the Middle: How Language Models Use Long Contexts"),
    ("foundations", "A Survey on Hallucination in Large Language Models"),
    # B. Core RAG
    ("core_rag", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"),
    ("core_rag", "REALM: Retrieval-Augmented Language Model Pre-Training"),
    ("core_rag", "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering"),
    ("core_rag", "Improving language models by retrieving from trillions of tokens"),
    ("core_rag", "Atlas: Few-shot Learning with Retrieval Augmented Language Models"),
    ("core_rag", "REPLUG: Retrieval-Augmented Black-Box Language Models"),
    ("core_rag", "In-Context Retrieval-Augmented Language Models"),
    ("core_rag", "Active Retrieval Augmented Generation"),
    ("core_rag", "Retrieval meets Long Context Large Language Models"),
    ("core_rag", "Retrieval-Augmented Generation for Large Language Models: A Survey"),
    # C. Advanced RAG
    ("advanced_rag", "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"),
    ("advanced_rag", "Corrective Retrieval Augmented Generation"),
    ("advanced_rag", "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity"),
    ("advanced_rag", "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"),
    ("advanced_rag", "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"),
    ("advanced_rag", "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models"),
    ("advanced_rag", "LightRAG: Simple and Fast Retrieval-Augmented Generation"),
    # D. Retrieval and embeddings
    ("retrieval_embeddings", "Dense Passage Retrieval for Open-Domain Question Answering"),
    ("retrieval_embeddings", "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT"),
    ("retrieval_embeddings", "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction"),
    ("retrieval_embeddings", "Unsupervised Dense Information Retrieval with Contrastive Learning"),
    ("retrieval_embeddings", "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking"),
    ("retrieval_embeddings", "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models"),
    ("retrieval_embeddings", "Text Embeddings by Weakly-Supervised Contrastive Pre-training"),
    ("retrieval_embeddings", "MTEB: Massive Text Embedding Benchmark"),
    ("retrieval_embeddings", "C-Pack"),
    ("retrieval_embeddings", "jina-embeddings-v3"),
    ("retrieval_embeddings", "ColPali: Efficient Document Retrieval with Vision Language Models"),
    # E. Query rewriting, chunking, reranking
    ("query_chunk_rerank", "Passage Re-ranking with BERT"),
    ("query_chunk_rerank", "Document Ranking with a Pretrained Sequence-to-Sequence Model"),
    ("query_chunk_rerank", "Precise Zero-Shot Dense Retrieval without Relevance Labels"),
    ("query_chunk_rerank", "Query2doc: Query Expansion with Large Language Models"),
    ("query_chunk_rerank", "Dense X Retrieval: What Retrieval Granularity Should We Use?"),
    ("query_chunk_rerank", "Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents"),
    ("query_chunk_rerank", "RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze!"),
    ("query_chunk_rerank", "Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models"),
    # F. Evaluation and failure analysis
    ("evaluation", "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"),
    ("evaluation", "RAGAs: Automated Evaluation of Retrieval Augmented Generation"),
    ("evaluation", "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems"),
    ("evaluation", "Benchmarking Large Language Models in Retrieval-Augmented Generation"),
    ("evaluation", "Evaluating Verifiability in Generative Search Engines"),
    ("evaluation", "Seven Failure Points When Engineering a Retrieval Augmented Generation System"),
    # G. Agentic and reasoning-based retrieval
    ("agentic", "ReAct: Synergizing Reasoning and Acting in Language Models"),
    ("agentic", "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG"),
    ("agentic", "Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning"),
]


def clean(text: str) -> str:
    """Lowercase and strip punctuation so titles compare fairly."""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def similarity(query: str, found: str) -> float:
    q, f = clean(query), clean(found)
    if q in f:  # our title is a part of the arXiv title -> good match
        return 1.0
    return SequenceMatcher(None, q, f).ratio()


def make_filename(authors, year, title) -> str:
    last_name = authors[0].name.split()[-1].lower()
    last_name = re.sub(r"[^a-z]", "", last_name)
    words = re.sub(r"[^a-z0-9 ]", " ", title.lower()).split()[:3]
    return f"{last_name}_{year}_{'_'.join(words)}.pdf"


def find_paper(client: arxiv.Client, title: str):
    """Try an exact-title query first, then a looser one."""
    queries = [f'ti:"{title}"', title]
    for q in queries:
        search = arxiv.Search(query=q, max_results=5, sort_by=arxiv.SortCriterion.Relevance)
        results = list(client.results(search))
        if results:
            best = max(results, key=lambda r: similarity(title, r.title))
            if similarity(title, best.title) >= MATCH_THRESHOLD:
                return best
    return None


def main():
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client(page_size=5, delay_seconds=3, num_retries=3)

    rows, not_found = [], []

    for i, (subtopic, title) in enumerate(PAPERS, start=1):
        print(f"[{i}/{len(PAPERS)}] {title[:70]}")
        try:
            paper = find_paper(client, title)
        except Exception as e:  # network hiccup etc.
            print(f"   ERROR: {e}")
            not_found.append(title)
            continue

        if paper is None:
            print("   NOT FOUND on arXiv")
            not_found.append(title)
            continue

        year = paper.published.year
        filename = make_filename(paper.authors, year, paper.title)
        pdf_path = PAPERS_DIR / filename

        if pdf_path.exists():
            print("   already downloaded, skipping")
        else:
            try:
                paper.download_pdf(dirpath=str(PAPERS_DIR), filename=filename)
                print(f"   saved -> {filename}")
                time.sleep(2)  # be polite to arXiv
            except Exception as e:
                print(f"   DOWNLOAD FAILED: {e}")
                not_found.append(title)
                continue

        rows.append(
            {
                "file_name": filename,
                "title": paper.title.replace("\n", " "),
                "authors": "; ".join(a.name for a in paper.authors[:5]),
                "year": year,
                "arxiv_link": paper.entry_id,
                "subtopic": subtopic,
            }
        )

    with open(METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file_name", "title", "authors", "year", "arxiv_link", "subtopic"]
        )
        writer.writeheader()
        writer.writerows(rows)

    NOT_FOUND_TXT.write_text("\n".join(not_found), encoding="utf-8")

    print("\n--- DONE ---")
    print(f"Downloaded / recorded: {len(rows)}")
    print(f"Not found or failed:   {len(not_found)}  (see {NOT_FOUND_TXT})")


if __name__ == "__main__":
    main()