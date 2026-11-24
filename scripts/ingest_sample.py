"""Ingest the sample docs into a Chroma collection.

Usage (from the repo root):
    python scripts/ingest_sample.py [--source data/sample_docs]
                                    [--collection agentic-rag-docs]
                                    [--persist-dir ./chroma_db]

Uses the same chunk-id / metadata schema as Project 1
(rag-data-ingestion-pipeline), so the collection is interchangeable
between the two repos.
"""
import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/ingest_sample.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.chunker import chunk_pages
from ingest.indexer import Indexer
from ingest.loader import load_folder
from retrieval.embedder import Embedder


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest sample docs into Chroma.")
    parser.add_argument("--source", default="data/sample_docs")
    parser.add_argument("--collection", default="agentic-rag-docs")
    parser.add_argument("--persist-dir", default="./chroma_db")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args()

    pages = load_folder(Path(args.source))
    print(f"Loaded {len(pages)} document(s) from {args.source}")
    if not pages:
        print("No .md files found — nothing to ingest.")
        return

    chunks = chunk_pages(pages, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"Produced {len(chunks)} chunks")

    embedder = Embedder()
    embeddings = embedder.embed([c.text for c in chunks])
    print(f"Embedded {len(embeddings)} chunks")

    indexer = Indexer(persist_dir=args.persist_dir)
    collection = indexer.get_collection(args.collection)
    added, duplicates = indexer.add(collection, chunks, embeddings)
    print(f"Indexed {added} chunks into '{args.collection}' "
          f"({duplicates} duplicates dropped)")


if __name__ == "__main__":
    main()
