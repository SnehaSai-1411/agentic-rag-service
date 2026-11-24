"""Smoke test: ingest the real sample docs into an ephemeral Chroma dir
with fake (keyword) embeddings, then run the agent end to end.

This exercises the real Chroma store and the real ingest code — only the
embeddings are fake, so the test stays offline and needs no model download.
"""
import re
from pathlib import Path

from agent.core import Agent
from config import Settings
from ingest.chunker import chunk_pages
from ingest.indexer import Indexer, chunk_id, content_hash
from ingest.loader import load_folder
from llm import ExtractiveFallback
from retrieval.store import ChromaRetriever
from tests.fakes import KeywordEmbedder

SAMPLE_DOCS = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


def build_indexed_retriever(tmp_path):
    pages = load_folder(SAMPLE_DOCS)
    assert len(pages) == 4  # rag-concepts, chunking, embeddings, vector-search
    chunks = chunk_pages(pages)
    assert len(chunks) >= 4

    embedder = KeywordEmbedder()
    embeddings = embedder.embed([c.text for c in chunks])

    indexer = Indexer(persist_dir=str(tmp_path / "chroma_db"))
    collection = indexer.get_collection("smoke-docs")
    added, duplicates = indexer.add(collection, chunks, embeddings)
    assert added == len(chunks)
    assert duplicates == 0

    # Project 1 schema check: ids and metadata match the shared format.
    stored = collection.get(include=["metadatas"])
    id_pattern = re.compile(r"^.+?-p\d+-c\d+-[0-9a-f]{12}$")
    for cid, meta in zip(stored["ids"], stored["metadatas"]):
        assert id_pattern.match(cid), f"bad chunk id: {cid}"
        assert set(meta) == {"source", "page", "chunk_index", "content_hash"}
        assert len(meta["content_hash"]) == 64
    first = chunks[0]
    assert chunk_id(first.source, first.page, first.chunk_index, first.text) \
        in stored["ids"]
    assert content_hash(first.text) in {
        m["content_hash"] for m in stored["metadatas"]}

    retriever = ChromaRetriever(
        embedder, persist_dir=str(tmp_path / "chroma_db"),
        collection_name="smoke-docs",
    )
    return retriever


def test_ingest_and_agent_end_to_end(tmp_path):
    retriever = build_indexed_retriever(tmp_path)
    agent = Agent(
        search_fn=retriever.search,
        llm=ExtractiveFallback(),
        settings=Settings(min_top_score=0.35, ambiguity_gap=0.05),
        embed_fn=retriever.embed_query,
    )

    resp = agent.ask("What is chunking and why does chunk size matter?")
    assert resp["decision"] == "answer", resp
    assert resp["citations"][0]["source"] == "chunking.md"
    assert "chunk" in resp["answer"].lower()
    assert resp["grounding"]["supported"] is True
    assert "Extractive answer" in resp["answer"]  # honestly labeled, no LLM key


def test_ingest_and_agent_dont_know_off_topic(tmp_path):
    retriever = build_indexed_retriever(tmp_path)
    agent = Agent(
        search_fn=retriever.search,
        llm=ExtractiveFallback(),
        settings=Settings(min_top_score=0.35, ambiguity_gap=0.05),
        embed_fn=retriever.embed_query,
    )
    resp = agent.ask("What is the capital of France?")
    assert resp["decision"] == "dont_know", resp
