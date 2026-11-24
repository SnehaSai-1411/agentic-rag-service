"""Turn text into embedding vectors (local model, no API key).

`sentence_transformers` is imported lazily (inside __init__) so importing
this module never loads torch — tests that inject a fake embedder stay fast
and offline.
"""
from collections.abc import Callable


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import

        # First run downloads the model (~90 MB) to a local cache.
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,   # cosine similarity behaves correctly
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


EmbedFn = Callable[[str], list[float]]
