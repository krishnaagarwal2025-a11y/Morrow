from src.database.retrieval import (
    search_similar_chunks,
    search_keyword_chunks,
    search_hybrid_rrf,
)
from src.embeddings.embedder import Embedder


QUERIES = [
    "MiniLM-L6-v2",
    "SQLite",
]


def print_results(title, results):
    print(f"\n{title}")
    print("-" * len(title))

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. "
            f"{result['filename']} | "
            f"chunk={result['chunk_id']} | "
            f"similarity={result.get('vector_score', result.get('similarity', 0.0)):.4f} | "
            f"keyword={result.get('keyword_score', 0.0):.4f} | "
            f"RRF={result.get('rrf_score', 0.0):.6f} | "
            f"vector_rank={result.get('vector_rank')} | "
            f"keyword_rank={result.get('keyword_rank')}"
        )


def main():
    embedder = Embedder()

    for query in QUERIES:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        query_embedding = embedder.embed_text(query)

        vector_results = search_similar_chunks(
            query_embedding,
            top_k=5
        )

        keyword_results = search_keyword_chunks(
            query,
            top_k=5
        )

        rrf_results = search_hybrid_rrf(
            query,
            embedder,
            top_k=5,
            candidate_k=5
        )

        print_results("VECTOR SEARCH", vector_results)
        print_results("KEYWORD SEARCH", keyword_results)
        print_results("RRF HYBRID SEARCH", rrf_results)


if __name__ == "__main__":
    main()