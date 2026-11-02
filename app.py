"""
app.py
Streamlit chat interface for the RAG research assistant.

Run from the project root:
    streamlit run app.py
"""

import csv
import re
from pathlib import Path

import streamlit as st

from src.rag_answer import RAGPipeline

st.set_page_config(page_title="RAG Research Assistant", page_icon="📚", layout="wide")

EXAMPLES = [
    "How does Self-RAG decide when to retrieve?",
    "What is HyDE and why does it help dense retrieval?",
    "Which algorithm does GraphRAG use to detect communities?",
    "How does ColBERTv2 reduce the storage cost of late interaction?",
    "Who won the 2022 FIFA World Cup?",
]


@st.cache_resource(show_spinner="Loading models (the first run takes about 30 seconds)...")
def load_pipeline():
    return RAGPipeline()


@st.cache_data
def load_links():
    """Map paper title -> arXiv link, from data/papers_metadata.csv."""
    links = {}
    path = Path("data/papers_metadata.csv")
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                links[" ".join(row["title"].lower().split())] = row["arxiv_link"]
    return links


def cited_numbers(answer: str):
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})


def source_block(n, s, links, show_passage_in_expander):
    link = links.get(" ".join(s["title"].lower().split()))
    header = f"**[{n}] {s['title']} ({s['year']})**"
    if link:
        header += f" - [arXiv]({link})"
    st.markdown(header)
    st.caption(f"Section: {s['section']}  |  rerank score {s['score']:.2f}")
    passage = " ".join(s["text"].split())
    if show_passage_in_expander:
        with st.expander("Show passage"):
            st.write(passage)
    else:  # Streamlit does not allow an expander inside an expander
        st.caption(passage[:400] + ("..." if len(passage) > 400 else ""))


def render_sources(sources, answer, links):
    total = len(sources)
    cited = [n for n in cited_numbers(answer) if 1 <= n <= total]
    others = [n for n in range(1, total + 1) if n not in cited]

    if cited:
        st.markdown("#### Cited sources")
        for n in cited:
            source_block(n, sources[n - 1], links, show_passage_in_expander=True)
    if others:
        label = "Other retrieved passages (not cited)" if cited else "Passages retrieved (none were cited)"
        with st.expander(label):
            for n in others:
                source_block(n, sources[n - 1], links, show_passage_in_expander=False)


# ---------- page ----------
rag = load_pipeline()
links = load_links()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    st.header("About")
    st.write(
        "Ask questions about 50 research papers on retrieval-augmented generation "
        "and retrieval for LLMs. Answers use only retrieved passages and are cited."
    )
    st.markdown(
        "**Pipeline:** vector search (top 20) -> cross-encoder rerank (top 5) -> LLM answer with citations."
    )
    st.subheader("Try an example")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state["pending"] = example
    if st.button("Clear chat"):
        st.session_state["messages"] = []
        st.rerun()
    st.caption(
        "Each question is answered independently (no chat memory). "
        "The language model runs on a free tier, so a rate-limit message means: wait a minute."
    )

st.title("📚 RAG Research Assistant")
st.caption("Answers are generated only from retrieved passages. Check the cited sources below each answer.")

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message["sources"], message["content"], links)

typed = st.chat_input("Ask about the papers...")
question = typed or st.session_state.pop("pending", None)

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the papers..."):
            result = rag.ask(question)
        st.markdown(result["answer"])
        render_sources(result["sources"], result["answer"], links)

    st.session_state["messages"].append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )