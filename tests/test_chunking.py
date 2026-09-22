from .chunker import chunk_document


files = [
    "sample.pdf",
    "sample.docx",
    "sample.txt",
    "sample.md"
]


for file in files:
    print("\n" + "=" * 60)
    print(f"FILE: {file}")
    print("=" * 60)

    chunks = chunk_document(
        file,
        chunk_size=100,
        overlap=20
    )

    print(f"Total chunks: {len(chunks)}")

    for chunk in chunks[:2]:
        print("\n--- Chunk ---")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Chunk index: {chunk['chunk_index']}")
        print(f"Word count: {chunk['word_count']}")
        print(f"Source: {chunk['source']}")
        print(f"File type: {chunk['file_type']}")
        print(f"Page: {chunk['page_number']}")
        print(f"Text: {chunk['text'][:200]}...")