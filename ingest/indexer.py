"""Index chunks into a persistent Chroma collection, dropping duplicates.

Chunk-id and metadata schema are identical to Project 1
(rag-data-ingestion-pipeline), so a collection built by either repo is
readable by the other:

- id:       "{source}-p{page}-c{chunk_index}-{sha256(text)[:12]}"
- metadata: source, page, chunk_index, content_hash (full sha256 hex)
- space:    cosine

`chromadb` is imported lazily (inside __init__) so importing this module
never loads the vector DB — tests that don't need Chroma stay fast.
"""
import hashlib


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_id(source: str, page: int, chunk_index: int, text: str) -> str:
    return f"{source}-p{page}-c{chunk_index}-{content_hash(text)[:12]}"


class Indexer:
    def __init__(self, persist_dir: str = "./chroma_db"):
        import chromadb  # lazy import

        self.client = chromadb.PersistentClient(path=persist_dir)

    def get_collection(self, name: str):
        return self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, collection, chunks, embeddings) -> tuple[int, int]:
        """Add (chunk, embedding) pairs. Returns (added, duplicates_dropped)."""
        seen: set[str] = set()
        ids, docs, metas, embs = [], [], [], []
        duplicates = 0
        for chunk, emb in zip(chunks, embeddings):
            h = content_hash(chunk.text)
            if h in seen:
                duplicates += 1
                continue
            seen.add(h)
            ids.append(chunk_id(chunk.source, chunk.page, chunk.chunk_index, chunk.text))
            docs.append(chunk.text)
            metas.append({
                "source": chunk.source,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "content_hash": h,
            })
            embs.append(emb)
        if ids:
            collection.upsert(ids=ids, documents=docs,
                              embeddings=embs, metadatas=metas)
        return len(ids), duplicates
