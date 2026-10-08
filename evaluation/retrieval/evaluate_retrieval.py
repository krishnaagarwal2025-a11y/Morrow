import json
import sys
from pathlib import Path

# Allow importing from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.embedder import Embedder
from src.database.retrieval import search_similar_chunks


QUESTIONS_FILE = Path(__file__).parent.parent / "datasets" / "questions.json"
if not QUESTIONS_FILE.exists():
    QUESTIONS_FILE = Path(__file__).parent / "questions.json"


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate():
    questions = load_questions()

    print("=" * 70)
    print("MORROW RETRIEVAL + EVIDENCE EVALUATION")
    print("=" * 70)

    print(f"\nTotal questions: {len(questions)}")

    embedder = Embedder()

    results = []

    for index, item in enumerate(questions, start=1):
        question = item["question"]
        expected_filename = item["expected_filename"]
        expected_evidence = item["expected_evidence"].lower()

        print(f"\n[{index}/{len(questions)}] {question}")

        query_embedding = embedder.embed_text(question)

        retrieved = search_similar_chunks(
            query_embedding,
            top_k=5,
            min_similarity=None
        )

        filenames = [
            result["filename"]
            for result in retrieved
        ]

        # Document-level retrieval
        hit_at_1 = expected_filename in filenames[:1]
        hit_at_3 = expected_filename in filenames[:3]
        hit_at_5 = expected_filename in filenames[:5]

        # Evidence-level retrieval
        evidence_at_1 = any(
            expected_evidence in result["text"].lower()
            for result in retrieved[:1]
        )

        evidence_at_3 = any(
            expected_evidence in result["text"].lower()
            for result in retrieved[:3]
        )

        evidence_at_5 = any(
            expected_evidence in result["text"].lower()
            for result in retrieved[:5]
        )

        results.append({
            "question": question,
            "document_at_1": hit_at_1,
            "document_at_3": hit_at_3,
            "document_at_5": hit_at_5,
            "evidence_at_1": evidence_at_1,
            "evidence_at_3": evidence_at_3,
            "evidence_at_5": evidence_at_5
        })

        print(f"Expected document: {expected_filename}")
        print(f"Expected evidence: {item['expected_evidence']}")

        print("\nRetrieved:")

        for rank, result in enumerate(retrieved, start=1):
            contains_evidence = (
                expected_evidence in result["text"].lower()
            )

            evidence_marker = (
                " <-- EVIDENCE"
                if contains_evidence
                else ""
            )

            print(
                f"  {rank}. "
                f"{result['filename']} "
                f"(similarity={result['similarity']:.4f})"
                f"{evidence_marker}"
            )

        print(
            f"\nDocument Recall@1: "
            f"{'PASS' if hit_at_1 else 'FAIL'}"
        )

        print(
            f"Document Recall@3: "
            f"{'PASS' if hit_at_3 else 'FAIL'}"
        )

        print(
            f"Document Recall@5: "
            f"{'PASS' if hit_at_5 else 'FAIL'}"
        )

        print(
            f"Evidence Recall@1: "
            f"{'PASS' if evidence_at_1 else 'FAIL'}"
        )

        print(
            f"Evidence Recall@3: "
            f"{'PASS' if evidence_at_3 else 'FAIL'}"
        )

        print(
            f"Evidence Recall@5: "
            f"{'PASS' if evidence_at_5 else 'FAIL'}"
        )

    total = len(results)

    # Document metrics
    document_recall_at_1 = sum(
        result["document_at_1"]
        for result in results
    ) / total

    document_recall_at_3 = sum(
        result["document_at_3"]
        for result in results
    ) / total

    document_recall_at_5 = sum(
        result["document_at_5"]
        for result in results
    ) / total

    # Evidence metrics
    evidence_recall_at_1 = sum(
        result["evidence_at_1"]
        for result in results
    ) / total

    evidence_recall_at_3 = sum(
        result["evidence_at_3"]
        for result in results
    ) / total

    evidence_recall_at_5 = sum(
        result["evidence_at_5"]
        for result in results
    ) / total

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print("\nDOCUMENT RETRIEVAL")
    print("-" * 70)

    print(
        f"Recall@1: "
        f"{document_recall_at_1 * 100:.2f}%"
    )

    print(
        f"Recall@3: "
        f"{document_recall_at_3 * 100:.2f}%"
    )

    print(
        f"Recall@5: "
        f"{document_recall_at_5 * 100:.2f}%"
    )

    print("\nEVIDENCE RETRIEVAL")
    print("-" * 70)

    print(
        f"Evidence Recall@1: "
        f"{evidence_recall_at_1 * 100:.2f}%"
    )

    print(
        f"Evidence Recall@3: "
        f"{evidence_recall_at_3 * 100:.2f}%"
    )

    print(
        f"Evidence Recall@5: "
        f"{evidence_recall_at_5 * 100:.2f}%"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    evaluate()