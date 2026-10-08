import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.embedder import Embedder


embedder = Embedder()


texts = [
    "PostgreSQL can store vector embeddings using pgvector.",
    "pgvector allows PostgreSQL to work with vector data.",
    "I went to the cafeteria and had lunch."
]


embeddings = embedder.embed_texts(texts)


for text, embedding in zip(texts, embeddings):

    print("\nText:")
    print(text)

    print("Vector dimensions:", len(embedding))
