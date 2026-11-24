"""Split page text into chunks that respect paragraph boundaries.

Same chunker as Project 1 (rag-data-ingestion-pipeline): paragraphs are kept
whole when they fit; a paragraph longer than chunk_size is hard-split with a
character overlap so context is not lost at the cut.
"""
from dataclasses import dataclass


@dataclass
class Chunk:
    source: str
    page: int
    chunk_index: int   # running index within the source document
    text: str


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split one page of text into chunks of at most ~chunk_size characters."""
    assert 0 <= overlap < chunk_size, "overlap must be smaller than chunk_size"
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunks.append(para[start:end])
                if end >= len(para):
                    break
                start = end - overlap
            continue
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return chunks


def chunk_pages(pages, chunk_size: int = 800, overlap: int = 100) -> list[Chunk]:
    chunks: list[Chunk] = []
    counters: dict[str, int] = {}
    for page in pages:
        if not page.text:
            continue   # empty pages produce no chunks
        for piece in chunk_text(page.text, chunk_size, overlap):
            idx = counters.get(page.source, 0)
            chunks.append(Chunk(source=page.source, page=page.page,
                                chunk_index=idx, text=piece))
            counters[page.source] = idx + 1
    return chunks
