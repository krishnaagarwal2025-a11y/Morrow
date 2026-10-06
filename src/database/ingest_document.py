import hashlib
from pathlib import Path

from ..document_ingestion.chunker import chunk_document
from ..embeddings.embedder import Embedder

from .connection import get_connection
from .repository import (
    insert_document,
    insert_chunk,
    find_document_by_hash
)


def calculate_file_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while chunk := file.read(8192):
            sha256.update(chunk)

    return sha256.hexdigest()


def ingest_document(file_path, original_filename=None):
    path = Path(file_path)

    filename = original_filename or path.name
    file_type = Path(filename).suffix.lower()

    print(f"Starting document ingestion: {filename}")

    # --------------------------------------------------
    # Calculate file hash
    # --------------------------------------------------

    file_hash = calculate_file_hash(file_path)

    print(f"File hash: {file_hash}")

    # --------------------------------------------------
    # Check for duplicate
    # --------------------------------------------------

    existing_document = find_document_by_hash(file_hash)

    if existing_document is not None:
        raise ValueError(
            f"Document already indexed: "
            f"{existing_document['filename']} "
            f"(Document ID: {existing_document['id']})"
        )

    # --------------------------------------------------
    # Extract and chunk document
    # --------------------------------------------------

    chunks = chunk_document(
        file_path,
        chunk_size=500,
        overlap=100
    )

    print(f"Created {len(chunks)} chunks.")

    if not chunks:
        raise ValueError(
            "No text could be extracted from the document."
        )

    # --------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------

    embedder = Embedder()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedder.embed_texts(texts)

    print(
        f"Generated {len(embeddings)} embeddings."
    )

    # --------------------------------------------------
    # Database transaction
    # --------------------------------------------------

    connection = get_connection()

    try:
        cursor = connection.cursor()

        # Insert document
        document_id = insert_document(
            cursor=cursor,
            filename=filename,
            file_path=str(path.resolve()),
            file_type=file_type,
            file_hash=file_hash
        )

        print(
            f"Inserted document with ID: {document_id}"
        )

        # Insert all chunks using the SAME cursor
        for chunk, embedding in zip(
            chunks,
            embeddings
        ):
            insert_chunk(
                cursor=cursor,
                document_id=document_id,
                chunk_index=chunk["chunk_index"],
                text=chunk["text"],
                page_number=chunk["page_number"],
                word_count=chunk["word_count"],
                embedding=embedding
            )

        # Commit everything together
        connection.commit()

        print(
            f"Inserted {len(chunks)} chunks into PostgreSQL."
        )

        print(
            "Document ingestion completed successfully."
        )

        return document_id

    except Exception:
        # Roll back document + chunks together
        connection.rollback()
        raise

    finally:
        connection.close()