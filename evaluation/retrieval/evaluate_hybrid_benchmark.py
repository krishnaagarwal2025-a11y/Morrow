import json
from pathlib import Path

from src.database.retrieval import (
    search_similar_chunks,
    search_keyword_chunks,
    search_hybrid_chunks
)
from src.embeddings.embedder import Embedder


TOP_K_VALUES = [1, 3, 5]


def load_questions():
    path = Path(__file__).parent.parent / "datasets" / "hybrid_questions.json"
    if not path.exists():
        path = Path(__file__).parent / "hybrid_questions.json"

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def check_result(results, expected_filename, expected_evidence, k):
    top_results = results[:k]

    document_found = any(
        result["filename"] == expected_filename
        for result in top_results
    )

    evidence_found = any(
        expected_evidence.lower() in result["text"].lower()
        for result in top_results
    )

    return document_found, evidence_found


def evaluate_method(
    questions,
    method_name,
    embedder
):
    document_hits = {
        k: 0
        for k in TOP_K_VALUES
    }

    evidence_hits = {
        k: 0
        for k in TOP_K_VALUES
    }

    total = len(questions)

    for item in questions:
        question = item["question"]
        expected_filename = item["expected_filename"]
        expected_evidence = item["expected_evidence"]

        if method_name == "Vector":
            query_embedding = embedder.embed_text(question)

            retrieved = search_similar_chunks(
                query_embedding,
                top_k=5
            )

        elif method_name == "Keyword":
            retrieved = search_keyword_chunks(
                question,
                top_k=5
            )

        elif method_name == "Hybrid":
            retrieved = search_hybrid_chunks(
                question,
                embedder,
                top_k=5
            )

        else:
            raise ValueError(
                f"Unknown method: {method_name}"
            )

        for k in TOP_K_VALUES:
            document_found, evidence_found = check_result(
                retrieved,
                expected_filename,
                expected_evidence,
                k
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


def print_results(results):
    print("\n" + "=" * 80)
    print("HYBRID RETRIEVAL BENCHMARK")
    print("=" * 80)

    print("\nDOCUMENT RECALL")
    print("-" * 80)

    print(
        f"{'Method':<12}"
        f"{'Recall@1':>12}"
        f"{'Recall@3':>12}"
        f"{'Recall@5':>12}"
    )

    print("-" * 80)

    for method, values in results.items():
        print(
            f"{method:<12}"
            f"{values['document'][1]:>11.2f}%"
            f"{values['document'][3]:>11.2f}%"
            f"{values['document'][5]:>11.2f}%"
        )

    print("\nEVIDENCE RECALL")
    print("-" * 80)

    print(
        f"{'Method':<12}"
        f"{'Evidence@1':>12}"
        f"{'Evidence@3':>12}"
        f"{'Evidence@5':>12}"
    )

    print("-" * 80)

    for method, values in results.items():
        print(
            f"{method:<12}"
            f"{values['evidence'][1]:>11.2f}%"
            f"{values['evidence'][3]:>11.2f}%"
            f"{values['evidence'][5]:>11.2f}%"
        )

    print("=" * 80)


def main():
    questions = load_questions()

    print(
        f"Evaluation dataset: "
        f"{len(questions)} questions"
    )

    print("\nLoading embedding model...")
    embedder = Embedder()
    print("Embedding model loaded.")

    results = {}

    for method in [
        "Vector",
        "Keyword",
        "Hybrid"
    ]:
        print(
            f"\nEvaluating {method} retrieval..."
        )

        results[method] = evaluate_method(
            questions,
            method,
            embedder
        )

    print_results(results)


if __name__ == "__main__":
    main()