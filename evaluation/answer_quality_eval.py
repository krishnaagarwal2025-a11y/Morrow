import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.rag_pipeline import RAGPipeline


pipeline = RAGPipeline(
    top_k=5,
    min_similarity=0.2554
)


TEST_CASES = [
    {
        "question": "How does Morrow use vector embeddings?",
        "expected": "answerable",
    },
    {
        "question": "What database does Morrow use?",
        "expected": "answerable",
    },
    {
        "question": "What information is stored about each chunk?",
        "expected": "answerable",
    },
    {
        "question": "What is the capital of France?",
        "expected": "unanswerable",
    },
    {
        "question": "Who founded Morrow?",
        "expected": "unanswerable",
    },
]


ABSTAIN_PHRASES = [
    "could not find enough information",
    "could not find relevant information",
]


def is_abstention(answer):
    answer_lower = answer.lower()

    return any(
        phrase in answer_lower
        for phrase in ABSTAIN_PHRASES
    )


def main():

    print("=" * 80)
    print("ANSWER QUALITY EVALUATION")
    print("=" * 80)

    answerable_pass = 0
    unanswerable_pass = 0

    for index, test in enumerate(TEST_CASES, start=1):

        question = test["question"]
        expected = test["expected"]

        print("\n" + "-" * 80)
        print(f"TEST {index}")
        print(f"Question: {question}")
        print(f"Expected: {expected}")

        result = pipeline.answer(question)

        answer = result["answer"]
        sources = result["sources"]

        print("\nAnswer:")
        print(answer)

        print("\nSources:")

        if not sources:
            print("  No sources")
        else:
            for source in sources:
                print(
                    f"  {source['filename']} | "
                    f"page={source['page_number']} | "
                    f"chunk={source['chunk_id']} | "
                    f"similarity={source['similarity']:.4f}"
                )

        # ---------------------------------------------------------
        # Answerable question
        # ---------------------------------------------------------

        if expected == "answerable":

            if sources and not is_abstention(answer):
                answerable_pass += 1
                print("\nResult: PASS")
            else:
                print("\nResult: FAIL")

        # ---------------------------------------------------------
        # Unanswerable question
        # ---------------------------------------------------------

        else:

            if is_abstention(answer):
                unanswerable_pass += 1
                print("\nResult: PASS")
            else:
                print("\nResult: FAIL")

    # -------------------------------------------------------------
    # Final metrics
    # -------------------------------------------------------------

    answerable_total = sum(
        1
        for test in TEST_CASES
        if test["expected"] == "answerable"
    )

    unanswerable_total = sum(
        1
        for test in TEST_CASES
        if test["expected"] == "unanswerable"
    )

    total_pass = answerable_pass + unanswerable_pass
    total_tests = len(TEST_CASES)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print(
        f"Answerable questions: "
        f"{answerable_pass}/{answerable_total}"
    )

    print(
        f"Unanswerable questions: "
        f"{unanswerable_pass}/{unanswerable_total}"
    )

    print(
        f"Overall evaluation: "
        f"{total_pass}/{total_tests} "
        f"({total_pass / total_tests:.2%})"
    )


if __name__ == "__main__":
    main()
