from src.database.retrieval import (
    search_similar_chunks,
    search_keyword_chunks
)
from src.embeddings.embedder import Embedder


def compare_query(query, embedder):
    print("\n" + "=" * 75)
    print(f"QUERY: {query}")
    print("=" * 75)

    query_embedding = embedder.embed_text(query)

    vector_results = search_similar_chunks(
        query_embedding,
        top_k=5
    )

    keyword_results = search_keyword_chunks(
        query,
        top_k=5
    )

    print("\nVECTOR RESULTS")
    print("-" * 75)

    for rank, result in enumerate(vector_results, start=1):
        print(
            f"Rank: {rank} | "
            f"Similarity: {result['similarity']:.4f} | "
            f"File: {result['filename']} | "
            f"Chunk: {result['chunk_id']}"
        )

    print("\nKEYWORD RESULTS")
    print("-" * 75)

    for rank, result in enumerate(keyword_results, start=1):
        print(
            f"Rank: {rank} | "
            f"Keyword score: {result['keyword_score']:.4f} | "
            f"File: {result['filename']} | "
            f"Chunk: {result['chunk_id']}"
        )

    if vector_results and keyword_results:
        print("\nTOP RESULT SCORE COMPARISON")
        print("-" * 75)

        vector_score = vector_results[0]["similarity"]
        keyword_score = keyword_results[0]["keyword_score"]

        print(f"Vector similarity : {vector_score:.4f}")
        print(f"Keyword score     : {keyword_score:.4f}")

        if keyword_score > vector_score:
            print("🔥 KEYWORD SCORE > VECTOR SCORE")
        else:
            print("Vector score >= keyword score")


def main():
    print("Loading embedding model...")
    embedder = Embedder()
    print("Embedding model loaded.")

    queries = [
        "SQLite",
        "MiniLM-L6-v2",
        "384",
        "300 words",
        "50 words",
        "4 chunks",
        "2024",
        "12,500 words",
        "180 milliseconds",
        "92 percent",
    ]

    for query in queries:
        compare_query(query, embedder)


if __name__ == "__main__":
    main()