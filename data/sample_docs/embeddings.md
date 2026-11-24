# Embeddings

An embedding is a dense vector of numbers that represents the meaning of a
piece of text. Texts with similar meanings get vectors that point in similar
directions, so similarity between texts can be measured with cosine
similarity between their vectors.

Embedding models are trained so that this geometric property holds. A
widely used local model is all-MiniLM-L6-v2, which produces 384-dimensional
vectors and runs on a CPU with no API key. It is small and fast, which makes
it a good default for a personal project, though larger models capture finer
distinctions.

Embeddings are usually normalized to unit length before they are stored.
With normalized vectors, cosine similarity reduces to a dot product, and
similarity scores stay comparable across queries.

One limitation is that a single vector must summarize a whole chunk. Long
or multi-topic chunks get a blended vector that may not match any single
question well. That is one more reason chunk size matters: the chunk and
the embedding are a package deal.
