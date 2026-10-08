import json
from pathlib import Path

from src.database.retrieval import search_hybrid_chunks
from src.embeddings.embedder import Embedder


def load_questions():
    questions_path = Path(__file__).parent / "datasets" / "questions.json"
    if not questions_path.exists():
        questions_path = Path(__file__).parent / "questions.json"

    with open(questions_path, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_hybrid_retrieval():
    questions = load_questions()

    results = {
        1: 0,
        3: 0,
        5: 0
    }

    evidence_results = {
        1: 0,
        3: 0,
        5: 0
    }

    total = len(questions)

    print("Loading embedding model...")
    embedder = Embedder()
    print("Embedding model loaded.")

    for item in questions:
        question = item["question"]
        expected_filename = item["expected_filename"]
        expected_evidence = item["expected_evidence"]

        retrieved = search_hybrid_chunks(
            question,
            embedder,
            top_k=5
        )

        for k in [1, 3, 5]:
            top_k_results = retrieved[:k]

            filenames = [
                result["filename"]
                for result in top_k_results
            ]

            evidence_found = any(
                expected_evidence.lower()
                in result["text"].lower()
                for result in top_k_results
            )

            if expected_filename in filenames:
                results[k] += 1

            if evidence_found:
                evidence_results[k] += 1

    print("\n" + "=" * 70)
    print("HYBRID RETRIEVAL EVALUATION")
    print("=" * 70)

    print("\nDOCUMENT RETRIEVAL")
    print("-" * 70)

    for k in [1, 3, 5]:
        recall = results[k] / total * 100

        print(
            f"Recall@{k}: "
            f"{recall:.2f}%"
        )

    print("\nEVIDENCE RETRIEVAL")
    print("-" * 70)

    for k in [1, 3, 5]:
        recall = evidence_results[k] / total * 100

        print(
            f"Evidence Recall@{k}: "
            f"{recall:.2f}%"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    evaluate_hybrid_retrieval()