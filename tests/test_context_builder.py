from ..embeddings.embedder import Embedder
from ..database.retrieval import search_similar_chunks
from .context_builder import build_context


embedder = Embedder()

query = "What database does the project use?"

query_embedding = embedder.embed_text(query)

results = search_similar_chunks(
    query_embedding,
    top_k=3
)

context = build_context(results)

print("\n" + "=" * 70)
print("QUESTION")
print("=" * 70)
print(query)

print("\n" + "=" * 70)
print("RETRIEVED CONTEXT")
print("=" * 70)
print(context)