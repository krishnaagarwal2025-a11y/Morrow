from src.database.retrieval import search_keyword_chunks


def test_query(query):
    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    results = search_keyword_chunks(
        query,
        top_k=5
    )

    if not results:
        print("NO KEYWORD RESULTS")
        return

    for rank, result in enumerate(results, start=1):
        print(
            f"\nRank: {rank}\n"
            f"File: {result['filename']}\n"
            f"Chunk: {result['chunk_id']}\n"
            f"Keyword score: {result['keyword_score']:.4f}\n"
            f"Text: {result['text'][:400]}"
        )


def main():
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
        test_query(query)


if __name__ == "__main__":
    main()