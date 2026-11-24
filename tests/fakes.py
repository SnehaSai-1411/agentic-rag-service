"""Shared test fakes: no model downloads, no API keys, no network."""
import math
import re


class FakeLLM:
    """Deterministic stand-in for a real LLM."""

    name = "fake-llm"

    def __init__(self, answer: str):
        self.answer = answer
        self.calls: list[tuple[str, list[str]]] = []

    def generate(self, question: str, context: list[str]) -> str:
        self.calls.append((question, context))
        return self.answer

    def generate_stream(self, question: str, context: list[str]):
        self.calls.append((question, context))
        for word in self.answer.split(" "):
            yield word + " "


def make_hits(specs):
    """Build retrieval hits from (text, source, score) triples."""
    hits = []
    for i, (text, source, score) in enumerate(specs):
        hits.append({
            "text": text,
            "source": source,
            "page": 1,
            "chunk_index": i,
            "score": score,
        })
    return hits


def fake_search(hits):
    """A search_fn that always returns the same hits."""

    def _search(question: str, k: int = 5):
        return [dict(h) for h in hits[:k]]

    return _search


# -- feature-hashing embeddings for the smoke test ---------------------------
# Deterministic, offline vectors: each word hashes to one of 64 dimensions
# and counts accumulate there. On-topic queries share words with the right
# document (high similarity); off-topic queries share only stopwords (low
# similarity). Coarse, but good enough to make retrieval — and the
# confidence gate — behave realistically without a model download.
#
# (An earlier keyword-count version failed here: with a per-dimension floor,
# an off-topic query became a near-uniform vector that matched every chunk
# at ~0.6, so the gate never said "don't know".)

import hashlib

_HASH_DIMS = 512

# Stopwords are excluded: without this, shared function words ("what", "is",
# "the") dominate the similarity and on-topic/off-topic queries score alike.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "for", "with", "by", "at", "from", "as", "and",
    "or", "but", "if", "then", "it", "its", "this", "that", "these", "those",
    "you", "your", "we", "our", "they", "their", "he", "she", "his", "her",
    "i", "me", "my", "do", "does", "did", "can", "could", "will", "would",
    "should", "so", "such", "than", "too", "very", "not", "no", "yes",
    "what", "why", "how", "when", "where", "which", "who",
}


class KeywordEmbedder:
    """Fake embedder: normalized feature-hashed word-count vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * _HASH_DIMS
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if word in _STOPWORDS:
                continue
            dim = int(hashlib.md5(word.encode()).hexdigest(), 16) % _HASH_DIMS
            vec[dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]


def word_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))
