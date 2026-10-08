import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_ingestion.chunker import chunk_document

base_dir = PROJECT_ROOT / "samples" if (PROJECT_ROOT / "samples").exists() else Path(".")

files = [
    str(base_dir / "sample.pdf"),
    str(base_dir / "sample.docx"),
    str(base_dir / "sample.txt"),
    str(base_dir / "sample.md")
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