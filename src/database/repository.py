from .connection import get_connection


def insert_document(
    cursor,
    filename,
    file_path,
    file_type,
    file_hash=None
):
    cursor.execute(
        """
        INSERT INTO documents (
            filename,
            file_path,
            file_type,
            file_hash
        )
        VALUES (%s, %s, %s, %s)
        RETURNING id;
        """,
        (
            filename,
            file_path,
            file_type,
            file_hash
        )
    )

    return cursor.fetchone()[0]


def find_document_by_hash(file_hash):
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT id, filename
            FROM documents
            WHERE file_hash = %s
            LIMIT 1;
            """,
            (file_hash,)
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return {
            "id": row[0],
            "filename": row[1]
        }

    finally:
        connection.close()


def insert_chunk(
    cursor,
    document_id,
    chunk_index,
    text,
    page_number,
    word_count,
    embedding
):
    cursor.execute(
        """
        INSERT INTO chunks (
            document_id,
            chunk_index,
            text,
            page_number,
            word_count,
            embedding
        )
        VALUES (%s, %s, %s, %s, %s, %s);
        """,
        (
            document_id,
            chunk_index,
            text,
            page_number,
            word_count,
            embedding.tolist()
        )
    )


def get_all_documents():
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                filename,
                file_type,
                created_at,
                file_hash
            FROM documents
            ORDER BY created_at DESC;
            """
        )

        rows = cursor.fetchall()

        documents = []

        for row in rows:
            documents.append({
                "id": row[0],
                "filename": row[1],
                "file_type": row[2],
                "created_at": row[3],
                "file_hash": row[4]
            })

        return documents

    finally:
        connection.close()


def delete_document(document_id):
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            DELETE FROM documents
            WHERE id = %s
            RETURNING filename;
            """,
            (document_id,)
        )

        row = cursor.fetchone()

        connection.commit()

        if row is None:
            return None

        return row[0]

    finally:
        connection.close()