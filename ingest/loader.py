"""Load Markdown files from a folder into per-page text records.

Mirrors Project 1's loader shape (rag-data-ingestion-pipeline) so the two
repos stay compatible: a Markdown file is treated as a single page (page=1),
since .md files have no page numbers. Citations therefore point at the
source file, not a page.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Page:
    source: str   # file name, e.g. "chunking.md"
    page: int     # always 1 for Markdown
    text: str


def load_markdown(path: Path) -> Page:
    return Page(source=path.name, page=1, text=path.read_text(encoding="utf-8").strip())


def load_folder(folder: Path) -> list[Page]:
    return [load_markdown(p) for p in sorted(folder.glob("*.md"))]
