from pathlib import Path

from src.database.connection import get_connection
from src.database.repository import find_document_by_path
from src.file_watcher.watcher import DocumentEventHandler

TEST_FOLDER = (
    Path(__file__).resolve().parents[1]
    / "documents"
    / "lifecycle_tests"
)


def get_chunk_count(document_id):
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM chunks
            WHERE document_id = %s;
            """,
            (document_id,)
        )

        return cursor.fetchone()[0]

    finally:
        connection.close()


def get_document_text(document_id):
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT text
            FROM chunks
            WHERE document_id = %s
            ORDER BY chunk_index;
            """,
            (document_id,)
        )

        rows = cursor.fetchall()

        return "\n".join(row[0] for row in rows)

    finally:
        connection.close()


def delete_test_document(document_id):
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            DELETE FROM documents
            WHERE id = %s;
            """,
            (document_id,)
        )

        connection.commit()

    finally:
        connection.close()


def test_document_lifecycle():
    TEST_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    original_file = TEST_FOLDER / "lifecycle_test.txt"
    renamed_file = TEST_FOLDER / "lifecycle_renamed.txt"

    handler = DocumentEventHandler(TEST_FOLDER)

    document_id = None

    try:
        # -------------------------------------------------
        # 1. CREATE
        # -------------------------------------------------

        original_file.write_text(
            "Morrow lifecycle test document. "
            "This is the original content.",
            encoding="utf-8"
        )

        handler.index_file(original_file)

        document = find_document_by_path(
            str(original_file.resolve())
        )

        assert document is not None
        assert document["filename"] == "lifecycle_test.txt"

        document_id = document["id"]

        assert get_chunk_count(document_id) > 0

        # -------------------------------------------------
        # 2. MODIFY
        # -------------------------------------------------

        original_file.write_text(
            "Morrow lifecycle test document. "
            "This is modified content.",
            encoding="utf-8"
        )

        handler.sync_modified_file(original_file)

        modified_document = find_document_by_path(
            str(original_file.resolve())
        )

        assert modified_document is not None

        modified_document_id = modified_document["id"]

        assert modified_document_id != document_id

        modified_text = get_document_text(
            modified_document_id
        )

        assert "modified content" in modified_text

        document_id = modified_document_id

        # -------------------------------------------------
        # 3. RENAME
        # -------------------------------------------------

        original_file.rename(renamed_file)

        class MoveEvent:
            is_directory = False
            src_path = str(original_file)
            dest_path = str(renamed_file)

        handler.on_moved(MoveEvent())

        renamed_document = find_document_by_path(
            str(renamed_file.resolve())
        )

        assert renamed_document is not None

        assert renamed_document["filename"] == (
            "lifecycle_renamed.txt"
        )

        assert renamed_document["id"] == document_id

        # -------------------------------------------------
        # 4. DELETE
        # -------------------------------------------------

        renamed_file.unlink()

        class DeleteEvent:
            is_directory = False
            src_path = str(renamed_file)

        handler.on_deleted(DeleteEvent())

        deleted_document = find_document_by_path(
            str(renamed_file.resolve())
        )

        assert deleted_document is None

        # -------------------------------------------------
        # 5. VERIFY CASCADE
        # -------------------------------------------------

        assert get_chunk_count(document_id) == 0

        document_id = None

    finally:
        # Safety cleanup
        if original_file.exists():
            original_file.unlink()

        if renamed_file.exists():
            renamed_file.unlink()

        if document_id is not None:
            delete_test_document(document_id)

        if TEST_FOLDER.exists():
            try:
                TEST_FOLDER.rmdir()
            except OSError:
                pass