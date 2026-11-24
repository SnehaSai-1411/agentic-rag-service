"""Semantic reranking: re-score retrieved chunks before answering.

Two modes:
- "cosine" (default): re-score each hit by cosine similarity between the
  query embedding and the hit's stored embedding. Cheap, no extra model,
  and usually just re-orders what the vector search already found — its
  value is mostly in cutting the list to the best few before the LLM sees it.
- "cross-encoder": score each (query, chunk) pair jointly with a
  cross-encoder model (ms-marco-MiniLM-L-6-v2). More accurate — the model
  reads the query and chunk together instead of comparing two independent
  vectors — but slower (one forward pass per pair) and needs a model
  download. Enabled with RERANKER=cross-encoder. Raw cross-encoder logits
  are sigmoid-normalized to [0, 1] so the confidence gate's thresholds keep
  their meaning in both modes.

If the cross-encoder cannot load (no network, no model cache), the reranker
falls back to cosine silently and records the fallback in the trace. If
neither can run (no embeddings on the hits), the original retrieval order
is kept.
"""
import math
from collections.abc import Callable


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    # float() keeps numpy scalars (from Chroma embeddings) out of the
    # trace and the JSON response: np.float32 is not JSON-serializable.
    return float(dot / (na * nb))


def cosine_rerank(
    query_embedding: list[float],
    hits: list[dict],
    top_n: int | None = None,
) -> list[dict]:
    """Re-score hits by cosine similarity to the query embedding.

    Hits carry their embedding under the "embedding" key. Hits without one
    keep their retrieval score and sink below re-scored hits. Returns new
    dicts with an updated "score" and a "rerank_score" field.
    """
    rescored: list[dict] = []
    for hit in hits:
        new_hit = dict(hit)
        emb = hit.get("embedding")
        if emb is not None:
            new_hit["rerank_score"] = cosine_similarity(query_embedding, emb)
            new_hit["score"] = new_hit["rerank_score"]
        else:
            new_hit["rerank_score"] = hit.get("score", 0.0)
        rescored.append(new_hit)
    rescored.sort(key=lambda h: h["rerank_score"], reverse=True)
    return rescored[:top_n] if top_n is not None else rescored


class Reranker:
    """Re-scores retrieved hits. The cross-encoder is loaded lazily and
    only when mode == "cross-encoder", so importing this module never
    touches torch."""

    def __init__(
        self,
        mode: str = "cosine",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        cross_encoder_factory: Callable[[str], object] | None = None,
    ):
        self.mode = mode
        self.cross_encoder_model = cross_encoder_model
        # Injectable factory so tests can simulate a load failure without
        # importing sentence-transformers at all.
        self._factory = cross_encoder_factory or self._default_factory
        self._cross_encoder = None
        self._cross_encoder_failed = False

    @staticmethod
    def _default_factory(model_name: str):
        from sentence_transformers import CrossEncoder  # lazy import

        return CrossEncoder(model_name)

    def _load_cross_encoder(self):
        if self._cross_encoder is None and not self._cross_encoder_failed:
            try:
                self._cross_encoder = self._factory(self.cross_encoder_model)
            except Exception:
                # No network / no cache / bad model name: fall back to cosine.
                self._cross_encoder_failed = True
        return self._cross_encoder

    def rerank(
        self,
        query: str,
        hits: list[dict],
        query_embedding: list[float] | None = None,
        top_n: int | None = None,
    ) -> tuple[list[dict], str]:
        """Return (reranked hits, mode actually used).

        `mode used` is one of "cross-encoder", "cosine", or "none" (original
        order kept because no re-scoring was possible).
        """
        if not hits:
            return hits, "none"
        if self.mode == "cross-encoder":
            model = self._load_cross_encoder()
            if model is not None:
                pairs = [(query, h["text"]) for h in hits]
                scores = model.predict(pairs)
                rescored = []
                for hit, score in zip(hits, scores):
                    new_hit = dict(hit)
                    # Cross-encoders output unbounded logits, but the gate
                    # expects scores in [0, 1] (see agent/gate.py). Sigmoid
                    # maps the logits onto that range so MIN_TOP_SCORE keeps
                    # working in both reranker modes.
                    new_hit["rerank_score"] = 1.0 / (1.0 + math.exp(-float(score)))
                    new_hit["score"] = new_hit["rerank_score"]
                    rescored.append(new_hit)
                rescored.sort(key=lambda h: h["rerank_score"], reverse=True)
                return (rescored[:top_n] if top_n is not None else rescored), "cross-encoder"
            # Fall through to cosine on load failure.
        if query_embedding is not None and any(
            h.get("embedding") is not None for h in hits
        ):
            return cosine_rerank(query_embedding, hits, top_n), "cosine"
        return (hits[:top_n] if top_n is not None else list(hits)), "none"
