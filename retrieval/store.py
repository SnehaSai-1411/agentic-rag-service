"""Retrieval over the local Chroma collection.

`ChromaRetriever.search_knowledge_base` is the agent's one tool: it embeds
the question, queries Chroma, and returns hits as plain dicts with cosine
*similarity* scores (Chroma returns cosine distances; similarity is
`1 - distance`, so higher is better everywhere downstream).

`chromadb` is imported lazily (inside __init__) so importing this module
never loads the vector DB.
"""
from ingest.indexer import Indexer


class ChromaRetriever:
    def __init__(
        self,
        embedder,
        persist_dir: str = "./chroma_db",
        collection_name: str = "agentic-rag-docs",
    ):
        self.embedder = embedder
        self.indexer = Indexer(persist_dir=persist_dir)
        self.collection = self.indexer.get_collection(collection_name)

    def embed_query(self, question: str) -> list[float]:
        return self.embedder.embed_one(question)

    def search(self, question: str, k: int = 5) -> list[dict]:
        return self.search_knowledge_base(question, k)

    def search_knowledge_base(self, question: str, k: int = 5) -> list[dict]:
        """The agent's tool: retrieve the top-k chunks for a question."""
        q_emb = self.embed_query(question)
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        hits = []
        docs = results.get("documents") or [[]]
        metas = results.get("metadatas") or [[]]
        dists = results.get("distances") or [[]]
        embs = results.get("embeddings") or [[]]
        for doc, meta, dist, emb in zip(docs[0], metas[0], dists[0], embs[0]):
            hits.append({
                "text": doc,
                "source": meta.get("source"),
                "page": meta.get("page"),
                "chunk_index": meta.get("chunk_index"),
                "score": 1.0 - float(dist),  # distance -> similarity
                # Cast to plain floats: Chroma returns numpy scalars, and
                # np.float32 is not JSON-serializable downstream.
                "embedding": [float(x) for x in emb] if emb is not None else None,
            })
        return hits
