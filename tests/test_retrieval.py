from ..embeddings.embedder import Embedder
from .retrieval import search_similar_chunks


embedder = Embedder()

query = "How will the system evaluate retrieval quality?"


query_embedding = embedder.embed_text(query)

results = search_similar_chunks(
    query_embedding,
    top_k=3
)

print("\nQuery:")
print(query)

print("\nTop relevant chunks:\n")

for result in results:
    print("-" * 60)
    print("Chunk ID:", result["chunk_id"])
    print("Document ID:", result["document_id"])
    print("Page:", result["page_number"])
    print("Similarity:", result["similarity"])
    print("Text:")
    print(result["text"])