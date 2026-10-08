import json
from pathlib import Path

from src.database.retrieval import search_keyword_chunks


def load_questions():
    questions_path = Path(__file__).parent.parent / "datasets" / "questions.json"
    if not questions_path.exists():
        questions_path = Path(__file__).parent / "questions.json"

    with open(questions_path, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_keyword_retrieval():
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

    for item in questions:
        question = item["question"]
        expected_filename = item["expected_filename"]
        expected_evidence = item["expected_evidence"]

        retrieved = search_keyword_chunks(
            question,
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

            if k == 5 and not evidence_found:
                print("\n" + "!" * 70)
                print("KEYWORD RETRIEVAL FAILURE")
                print("!" * 70)
                print(f"Question: {question}")
                print(f"Expected file: {expected_filename}")
                print(f"Expected evidence: {expected_evidence}")

                print("\nRetrieved results:")

                for result in retrieved:
                    print(
                        f"\n"
                        f"File: {result['filename']}\n"
                        f"Chunk: {result['chunk_id']}\n"
                        f"Keyword score: {result['keyword_score']:.4f}\n"
                        f"Text: {result['text'][:300]}"
                    )

    print("=" * 70)
    print("KEYWORD RETRIEVAL EVALUATION")
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
    evaluate_keyword_retrieval()