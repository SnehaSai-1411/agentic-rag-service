# Retrieval-Augmented Generation (RAG)

Retrieval-augmented generation is a pattern for question answering that
combines a retriever with a language model. Instead of asking the model to
answer from its training data alone, the system first retrieves relevant
passages from a knowledge base and then asks the model to answer using only
those passages.

The retriever is usually a vector search over embedded document chunks. A
user question is embedded with the same model that embedded the documents,
and the nearest chunks by cosine similarity are returned as context.

RAG helps in two ways. First, it grounds the answer in specific documents,
which reduces hallucination: the model has source text to work from instead
of guessing. Second, it lets the knowledge base change without retraining
the model — new documents are indexed and immediately become answerable.

The trade-off is that answer quality now depends on retrieval quality. If
the retriever returns irrelevant chunks, the model will either answer from
the wrong context or, if it is well instructed, refuse to answer. That is
why production RAG systems add reranking, confidence thresholds, and
grounding checks on top of plain retrieval.
