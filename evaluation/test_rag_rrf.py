from src.rag.rag_pipeline import RAGPipeline


def main():
    pipeline = RAGPipeline(top_k=5)

    question = "What database does Aurora Notes use?"

    result = pipeline.answer(question)

    print("\nQUESTION")
    print("--------")
    print(result["question"])

    print("\nANSWER")
    print("------")
    print(result["answer"])

    print("\nSOURCES")
    print("-------")
    for source in result["sources"]:
        print(source)

    print("\nRETRIEVED CONTEXT")
    print("-----------------")
    print(result["retrieved_context"])


if __name__ == "__main__":
    main()