"""Retrieval package: local embeddings and Chroma search."""
from retrieval.embedder import Embedder
from retrieval.store import ChromaRetriever

__all__ = ["Embedder", "ChromaRetriever"]
