"""Tests for the reranker — pure cosine math plus a load-failure fallback."""
from agent.rerank import Reranker, cosine_rerank, cosine_similarity


def test_cosine_similarity_basics():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0
    assert cosine_similarity([1, 0], [0, 0]) == 0.0


def test_cosine_rerank_reorders_a_known_set():
    # Retrieval ranked the off-topic chunk first; cosine re-scoring fixes it.
    query_emb = [1.0, 0.0]
    hits = [
        {"text": "off-topic", "source": "a.md", "score": 0.95,
         "embedding": [0.0, 1.0]},
        {"text": "on-topic", "source": "b.md", "score": 0.50,
         "embedding": [1.0, 0.0]},
        {"text": "half-related", "source": "c.md", "score": 0.70,
         "embedding": [0.7071, 0.7071]},
    ]
    reranked = cosine_rerank(query_emb, hits)
    assert [h["text"] for h in reranked] == [
        "on-topic", "half-related", "off-topic"]
    assert reranked[0]["rerank_score"] == 1.0


def test_cosine_rerank_top_n_truncates():
    query_emb = [1.0, 0.0]
    hits = [
        {"text": "a", "score": 0.1, "embedding": [1.0, 0.0]},
        {"text": "b", "score": 0.2, "embedding": [0.0, 1.0]},
    ]
    assert len(cosine_rerank(query_emb, hits, top_n=1)) == 1


def test_cosine_rerank_keeps_hits_without_embeddings():
    query_emb = [1.0, 0.0]
    hits = [
        {"text": "no-embedding", "score": 0.9},
        {"text": "with-embedding", "score": 0.5, "embedding": [1.0, 0.0]},
    ]
    reranked = cosine_rerank(query_emb, hits)
    assert reranked[0]["text"] == "with-embedding"
    assert reranked[1]["text"] == "no-embedding"


def test_cross_encoder_falls_back_to_cosine_when_model_cannot_load():
    def failing_factory(model_name):
        raise RuntimeError("no network, no model cache")

    reranker = Reranker(mode="cross-encoder",
                        cross_encoder_factory=failing_factory)
    query_emb = [1.0, 0.0]
    hits = [
        {"text": "off-topic", "score": 0.95, "embedding": [0.0, 1.0]},
        {"text": "on-topic", "score": 0.50, "embedding": [1.0, 0.0]},
    ]
    reranked, mode_used = reranker.rerank("q", hits, query_embedding=query_emb)
    assert mode_used == "cosine"  # graceful fallback, not an exception
    assert reranked[0]["text"] == "on-topic"


def test_cross_encoder_used_when_factory_succeeds():
    class FakeCrossEncoder:
        def predict(self, pairs):
            # Score by whether the chunk mentions "chunking".
            return [1.0 if "chunking" in text else 0.0
                    for _, text in pairs]

    reranker = Reranker(mode="cross-encoder",
                        cross_encoder_factory=lambda name: FakeCrossEncoder())
    hits = [
        {"text": "about vectors", "score": 0.95},
        {"text": "about chunking", "score": 0.50},
    ]
    reranked, mode_used = reranker.rerank("what is chunking", hits)
    assert mode_used == "cross-encoder"
    assert reranked[0]["text"] == "about chunking"


def test_rerank_keeps_order_when_nothing_can_rescore():
    reranker = Reranker(mode="cosine")
    hits = [{"text": "a", "score": 0.9}, {"text": "b", "score": 0.8}]
    reranked, mode_used = reranker.rerank("q", hits, query_embedding=None)
    assert mode_used == "none"
    assert [h["text"] for h in reranked] == ["a", "b"]
