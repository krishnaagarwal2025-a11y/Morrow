import json
from pathlib import Path

from src.database.retrieval import (
    search_similar_chunks,
    search_keyword_chunks,
    search_hybrid_rrf,
)
from src.embeddings.embedder import Embedder


DATASET_PATH = Path(__file__).parent.parent / "datasets" / "hybrid_questions.json"
if not DATASET_PATH.exists():
    DATASET_PATH = Path("evaluation/hybrid_questions.json")


def load_questions():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_results(results, question):
    expected_filename = question["expected_filename"]
    expected_evidence = question["expected_evidence"]

    filenames = [result["filename"] for result in results]
    evidence = [result["text"] for result in results]

    document_found = expected_filename in filenames

    evidence_found = any(
        expected_evidence in text
        for text in evidence
    )

    return document_found, evidence_found


def main():
    questions = load_questions()

    print(f"Evaluation dataset: {len(questions)} questions")

    embedder = Embedder()

    methods = {
        "Vector": [],
        "Keyword": [],
        "RRF": [],
    }

    for question in questions:
        query = question["question"]

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

        methods["Vector"].append(vector_results)
        methods["Keyword"].append(keyword_results)
        methods["RRF"].append(rrf_results)

    print("\nDOCUMENT RECALL")
    print("=" * 60)

    for method, all_results in methods.items():

        recalls = []

        for k in [1, 3, 5]:
            correct = 0

            for question, results in zip(
                questions,
                all_results
            ):
                found, _ = evaluate_results(
                    results[:k],
                    question
                )

                if found:
                    correct += 1

            recall = correct / len(questions) * 100
            recalls.append(recall)

        print(
            f"{method:<10}"
            f"Recall@1: {recalls[0]:6.2f}%   "
            f"Recall@3: {recalls[1]:6.2f}%   "
            f"Recall@5: {recalls[2]:6.2f}%"
        )

    print("\nEVIDENCE RECALL")
    print("=" * 60)

    for method, all_results in methods.items():

        recalls = []

        for k in [1, 3, 5]:
            correct = 0

            for question, results in zip(
                questions,
                all_results
            ):
                _, evidence_found = evaluate_results(
                    results[:k],
                    question
                )

                if evidence_found:
                    correct += 1

            recall = correct / len(questions) * 100
            recalls.append(recall)

        print(
            f"{method:<10}"
            f"Evidence@1: {recalls[0]:6.2f}%   "
            f"Evidence@3: {recalls[1]:6.2f}%   "
            f"Evidence@5: {recalls[2]:6.2f}%"
        )


if __name__ == "__main__":
    main()