from src.database.retrieval import search_hybrid_chunks
from src.embeddings.embedder import Embedder


def test_query(query, embedder):
    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    results = search_hybrid_chunks(
        query,
        embedder,
        top_k=5
    )

    for rank, result in enumerate(results, start=1):
        print(
            f"\n"
            f"Rank: {rank}\n"
            f"File: {result['filename']}\n"
            f"Chunk: {result['chunk_id']}\n"
            f"Vector score: {result['vector_score']:.4f}\n"
            f"Keyword score: {result['keyword_score']:.4f}\n"
            f"Hybrid score: {result['hybrid_score']:.4f}\n"
            f"Text: {result['text'][:300]}"
        )


def main():
    print("Loading embedding model...")
    embedder = Embedder()
    print("Embedding model loaded.")

    test_query(
        "What database does Aurora Notes use?",
        embedder
    )

    test_query(
        "How many chunks are retrieved for each question?",
        embedder
    )


if __name__ == "__main__":
    main()