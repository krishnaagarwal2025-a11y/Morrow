import json
from pathlib import Path

from src.database.retrieval import search_hybrid_chunks
from src.embeddings.embedder import Embedder


TOP_K_VALUES = [1, 3, 5]

WEIGHTS = [
    (1.0, 0.0),
    (0.9, 0.1),
    (0.8, 0.2),
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
    (0.3, 0.7),
    (0.2, 0.8),
    (0.1, 0.9),
    (0.0, 1.0),
]


def load_questions():
    path = Path(__file__).parent / "datasets" / "hybrid_questions.json"
    if not path.exists():
        path = Path(__file__).parent / "hybrid_questions.json"

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_weights(
    questions,
    embedder,
    vector_weight,
    keyword_weight
):
    document_hits = {k: 0 for k in TOP_K_VALUES}
    evidence_hits = {k: 0 for k in TOP_K_VALUES}

    total = len(questions)

    for item in questions:
        question = item["question"]
        expected_filename = item["expected_filename"]
        expected_evidence = item["expected_evidence"]

        results = search_hybrid_chunks(
            question,
            embedder,
            top_k=5,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight
        )

        for k in TOP_K_VALUES:
            top_results = results[:k]

            document_found = any(
                result["filename"] == expected_filename
                for result in top_results
            )

            evidence_found = any(
                expected_evidence.lower()
                in result["text"].lower()
                for result in top_results
            )

            if document_found:
                document_hits[k] += 1

            if evidence_found:
                evidence_hits[k] += 1

    return {
        "document": {
            k: document_hits[k] / total * 100
            for k in TOP_K_VALUES
        },
        "evidence": {
            k: evidence_hits[k] / total * 100
            for k in TOP_K_VALUES
        }
    }


def main():
    questions = load_questions()

    print(
        f"Evaluation dataset: "
        f"{len(questions)} questions"
    )

    print("\nLoading embedding model...")
    embedder = Embedder()
    print("Embedding model loaded.")

    results = []

    print("\nRunning weight sweep...")

    for vector_weight, keyword_weight in WEIGHTS:
        print(
            f"Testing Vector={vector_weight:.1f}, "
            f"Keyword={keyword_weight:.1f}"
        )

        metrics = evaluate_weights(
            questions,
            embedder,
            vector_weight,
            keyword_weight
        )

        results.append({
            "vector_weight": vector_weight,
            "keyword_weight": keyword_weight,
            **metrics
        })

    print("\n" + "=" * 90)
    print("HYBRID WEIGHT SWEEP")
    print("=" * 90)

    print(
        f"{'Vector':>8}"
        f"{'Keyword':>10}"
        f"{'Recall@1':>12}"
        f"{'Recall@3':>12}"
        f"{'Recall@5':>12}"
        f"{'Evidence@1':>14}"
        f"{'Evidence@3':>14}"
        f"{'Evidence@5':>14}"
    )

    print("-" * 90)

    for result in results:
        print(
            f"{result['vector_weight']:>8.1f}"
            f"{result['keyword_weight']:>10.1f}"
            f"{result['document'][1]:>11.2f}%"
            f"{result['document'][3]:>11.2f}%"
            f"{result['document'][5]:>11.2f}%"
            f"{result['evidence'][1]:>13.2f}%"
            f"{result['evidence'][3]:>13.2f}%"
            f"{result['evidence'][5]:>13.2f}%"
        )

    print("=" * 90)


if __name__ == "__main__":
    main()