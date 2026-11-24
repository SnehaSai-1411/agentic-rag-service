# Chunking Strategies for RAG

Chunking is the step that splits documents into the pieces a retriever can
return. The size of a chunk is a trade-off. Small chunks are precise: a
retrieved chunk is likely to be exactly about the question. But a small
chunk can cut a sentence or an argument in half, leaving the language model
without enough context to answer well.

Large chunks carry more context, but they dilute relevance. A thousand-word
chunk may contain the answer buried among unrelated paragraphs, and the
embedding of the whole chunk reflects the average of everything in it, not
the one relevant sentence.

A common default is around 800 characters per chunk with about 100
characters of overlap between consecutive chunks. The overlap exists so
that a sentence straddling a boundary still appears whole in at least one
chunk.

Paragraph-aware chunking improves on fixed-size splitting: it keeps
paragraphs whole when they fit, and only hard-splits a paragraph that is
longer than the chunk size. Semantic units stay intact, which makes both
the embedding and the retrieved text more coherent.
