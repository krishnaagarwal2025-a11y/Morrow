import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.embedder import Embedder
from src.database.retrieval import search_similar_chunks
from src.rag.context_builder import build_context


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
