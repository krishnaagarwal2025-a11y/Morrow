from .connection import get_connection


def search_similar_chunks(
    query_embedding,
    top_k=5,
    min_similarity=None
):
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    connection = get_connection()

    try:
        cursor = connection.cursor()

        if min_similarity is None:

            cursor.execute(
                """
                SELECT
                    chunks.id,
                    chunks.document_id,
                    documents.filename,
                    chunks.chunk_index,
                    chunks.text,
                    chunks.page_number,
                    chunks.word_count,
                    1 - (chunks.embedding <=> %s) AS similarity
                FROM chunks
                JOIN documents
                    ON chunks.document_id = documents.id
                WHERE chunks.embedding IS NOT NULL
                ORDER BY chunks.embedding <=> %s
                LIMIT %s;
                """,
                (
                    query_embedding,
                    query_embedding,
                    top_k
                )
            )

        else:

            cursor.execute(
                """
                SELECT
                    chunks.id,
                    chunks.document_id,
                    documents.filename,
                    chunks.chunk_index,
                    chunks.text,
                    chunks.page_number,
                    chunks.word_count,
                    1 - (chunks.embedding <=> %s) AS similarity
                FROM chunks
                JOIN documents
                    ON chunks.document_id = documents.id
                WHERE
                    chunks.embedding IS NOT NULL
                    AND 1 - (chunks.embedding <=> %s) >= %s
                ORDER BY chunks.embedding <=> %s
                LIMIT %s;
                """,
                (
                    query_embedding,
                    query_embedding,
                    min_similarity,
                    query_embedding,
                    top_k
                )
            )

        rows = cursor.fetchall()

        results = []

        for row in rows:
            results.append({
                "chunk_id": row[0],
                "document_id": row[1],
                "filename": row[2],
                "chunk_index": row[3],
                "text": row[4],
                "page_number": row[5],
                "word_count": row[6],
                "similarity": float(row[7])
            })

        return results

    finally:
        connection.close()