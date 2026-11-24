# Vector Search

Vector search finds the stored vectors nearest to a query vector. For small
collections, an exact search that scores every vector is fine. For large
collections, approximate nearest neighbor (ANN) indexes trade a little
accuracy for much faster queries.

HNSW (Hierarchical Navigable Small World) is a common ANN algorithm. It
builds a layered graph over the vectors: the top layers are sparse and let
a search jump quickly toward the right neighborhood, and the bottom layer
holds every vector for a fine-grained final comparison.

The distance metric matters. Cosine distance measures the angle between
vectors and ignores their length, which suits text embeddings — two texts
about the same topic should match even if one is longer. Euclidean distance
measures straight-line distance and is sensitive to vector length, so it is
usually a worse fit for text unless vectors are normalized first.

Chroma is a lightweight vector database that stores embeddings with their
source text and metadata, and supports cosine search out of the box. It runs
embedded in the application process, which makes it a zero-setup choice for
a local RAG service.
