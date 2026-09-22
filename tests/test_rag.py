from .rag_pipeline import RAGPipeline


pipeline = RAGPipeline(
    top_k=5,
    min_similarity=0.2554
)


test_queries = [
    {
        "question": "How are document chunks converted into embeddings?",
        "expected": "answerable"
    },
    {
        "question": "How does Morrow convert chunks into vectors?",
        "expected": "answerable"
    },
    {
        "question": "How are chunks represented as vectors?",
        "expected": "answerable"
    }
]


print("\n" + "=" * 80)
print("MORROW RELEVANCE GATE TEST")
print("=" * 80)


for index, test in enumerate(test_queries, start=1):

    question = test["question"]
    expected = test["expected"]

    result = pipeline.answer(question)

    sources = result["sources"]

    if sources:
        top_similarity = max(
            source["similarity"]
            for source in sources
        )
    else:
        top_similarity = None

    print("\n" + "-" * 80)
    print(f"TEST {index}")
    print("-" * 80)

    print(f"Question: {question}")
    print(f"Expected: {expected}")

    if top_similarity is not None:
        print(f"Top similarity: {top_similarity:.4f}")
    else:
        print("Top similarity: None")

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")

    if sources:
        for source in sources:
            print(
                f"File: {source['filename']} | "
                f"Page: {source['page_number']} | "
                f"Chunk: {source['chunk_id']} | "
                f"Similarity: {source['similarity']:.4f}"
            )
    else:
        print("No relevant sources found.")


print("\n" + "=" * 80)
print("RELEVANCE GATE TEST COMPLETE")
print("=" * 80)