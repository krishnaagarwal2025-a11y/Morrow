import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.retrieval import search_keyword_chunks


def main():
    queries = [
        "SQLite",
        "MiniLM-L6-v2",
        "384",
        "retrieval accuracy"
    ]

    for query in queries:
        print("\n" + "=" * 60)
        print(f"QUERY: {query}")
        print("=" * 60)

        results = search_keyword_chunks(
            query,
            top_k=5
        )

        for result in results:
            print(
                f"\n"
                f"File: {result['filename']}\n"
                f"Chunk: {result['chunk_id']}\n"
                f"Keyword score: {result['keyword_score']:.4f}\n"
                f"Text: {result['text'][:300]}"
            )


if __name__ == "__main__":
    main()
