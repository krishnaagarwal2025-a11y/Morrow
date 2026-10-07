import time
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from ..database.ingest_document import (
    ingest_document,
    calculate_file_hash
)

from ..database.repository import (
    find_document_by_path,
    delete_document_by_path
)

from ..database.connection import get_connection
from ..database.repository import find_document_by_path


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md"
}


class DocumentEventHandler(FileSystemEventHandler):


    def on_deleted(self, event):
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        if not self._is_supported_file(file_path):
            return

        print(
            f"\nDocument deleted: "
            f"{file_path.name}"
        )

        try:
            resolved_path = str(file_path.resolve())

            deleted_document = delete_document_by_path(
                resolved_path
            )

            if deleted_document is None:
                print(
                    f"No database record found for "
                    f"{file_path.name}. Nothing to remove."
                )
                return

            print(
                f"Removed {deleted_document['filename']} "
                f"from Morrow "
                f"(Document ID: {deleted_document['id']})"
            )

        except Exception as error:
            print(
                f"Failed to remove deleted document "
                f"{file_path.name}: {error}"
            )

    def _is_supported_file(self, file_path):
        return (
            Path(file_path).suffix.lower()
            in SUPPORTED_EXTENSIONS
        )

    def _is_file_ready(self, file_path):
        path = Path(file_path)

        return (
            path.exists()
            and path.is_file()
            and path.stat().st_size > 0
        )

    def _delete_document(self, document_id):
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

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    def on_created(self, event):
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        if not self._is_supported_file(file_path):
            return

        if not self._is_file_ready(file_path):
            print(
                f"Skipped {file_path.name}: "
                "file is empty or unavailable."
            )
            return

        print(
            f"\nNew document detected: "
            f"{file_path.name}"
        )

        try:
            document_id = ingest_document(
                str(file_path),
                original_filename=file_path.name
            )

            print(
                f"Successfully indexed "
                f"{file_path.name} "
                f"(Document ID: {document_id})"
            )

        except ValueError as error:
            print(
                f"Skipped {file_path.name}: {error}"
            )

        except Exception as error:
            print(
                f"Failed to index "
                f"{file_path.name}: {error}"
            )

    def on_modified(self, event):
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        if not self._is_supported_file(file_path):
            return

        if not self._is_file_ready(file_path):
            return

        print(
            f"\nDocument modified: "
            f"{file_path.name}"
        )

        try:
            resolved_path = str(file_path.resolve())

            existing_document = find_document_by_path(
                resolved_path
            )

            if existing_document is None:
                print(
                    "No existing database record found. "
                    "Indexing as a new document."
                )

                document_id = ingest_document(
                    str(file_path),
                    original_filename=file_path.name
                )

                print(
                    f"Successfully indexed "
                    f"{file_path.name} "
                    f"(Document ID: {document_id})"
                )

                return

            new_hash = calculate_file_hash(
                str(file_path)
            )

            if new_hash == existing_document["file_hash"]:
                print(
                    f"No content change detected "
                    f"for {file_path.name}. "
                    "Skipping re-indexing."
                )
                return

            print(
                f"Content changed for "
                f"{file_path.name}. "
                f"Removing old version..."
            )

            self._delete_document(
                existing_document["id"]
            )

            document_id = ingest_document(
                str(file_path),
                original_filename=file_path.name
            )

            print(
                f"Successfully re-indexed "
                f"{file_path.name} "
                f"(Document ID: {document_id})"
            )

        except ValueError as error:
            print(
                f"Skipped modified file "
                f"{file_path.name}: {error}"
            )

        except Exception as error:
            print(
                f"Failed to process modified file "
                f"{file_path.name}: {error}"
            )


def start_watcher(folder_path):
    folder = Path(folder_path)

    if not folder.exists():
        folder.mkdir(
            parents=True,
            exist_ok=True
        )

    if not folder.is_dir():
        raise ValueError(
            f"Watcher path is not a directory: {folder}"
        )

    event_handler = DocumentEventHandler()

    observer = Observer()

    observer.schedule(
        event_handler,
        str(folder),
        recursive=False
    )

    observer.start()

    print("=" * 60)
    print("MORROW DOCUMENT WATCHER")
    print("=" * 60)
    print(f"Watching: {folder.resolve()}")
    print(
        "Supported: "
        ".pdf, .docx, .txt, .md"
    )
    print("\nWaiting for document changes...")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping watcher...")
        observer.stop()

    observer.join()

    print("Watcher stopped.")


if __name__ == "__main__":
    documents_folder = (
        Path(__file__).resolve().parents[2]
        / "documents"
    )

    start_watcher(documents_folder)