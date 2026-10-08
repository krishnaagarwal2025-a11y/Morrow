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
        "question": "What database does Morrow use?",
        "expected_page": 1,
        "expected_keyword": "PostgreSQL",
    },
    {
        "question": "What information is stored about each chunk?",
        "expected_page": 3,
        "expected_keyword": "Chunk ID",
    },
    {
        "question": "How does the retrieval layer work?",
        "expected_page": 4,
        "expected_keyword": "embedding",
    },
]


def main():

    print("=" * 80)
    print("CITATION ACCURACY EVALUATION")
    print("=" * 80)

    passed = 0

    for index, test in enumerate(TEST_CASES, start=1):

        question = test["question"]
        expected_page = test["expected_page"]
        expected_keyword = test["expected_keyword"]

        print("\n" + "-" * 80)
        print(f"TEST {index}")
        print(f"Question: {question}")
        print(f"Expected page: {expected_page}")
        print(f"Expected keyword: {expected_keyword}")

        result = pipeline.answer(question)

        print("\nAnswer:")
        print(result["answer"])

        print("\nSources:")

        citation_found = False
        supporting_source_found = False

        for source in result["sources"]:

            print(
                f"  {source['filename']} | "
                f"page={source['page_number']} | "
                f"chunk={source['chunk_id']} | "
                f"similarity={source['similarity']:.4f}"
            )

            if source["page_number"] == expected_page:
                citation_found = True

                # Retrieve the actual chunk text from the result
                # corresponding to this source.
                #
                # The source formatter currently exposes metadata,
                # so this test checks page-level citation presence.

        # ---------------------------------------------------------
        # Citation evaluation
        # ---------------------------------------------------------

        if citation_found:
            supporting_source_found = True

        if supporting_source_found:
            passed += 1
            print("\nResult: PASS")
        else:
            print("\nResult: FAIL")

    # -------------------------------------------------------------
    # Final result
    # -------------------------------------------------------------

    total = len(TEST_CASES)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print(f"Citation tests passed: {passed}/{total}")
    print(f"Citation Accuracy: {passed / total:.2%}")


if __name__ == "__main__":
    main()
