from .embedder import Embedder


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