import time
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from ..database.ingest_document import (
    ingest_document,
    calculate_file_hash
)

from ..database.connection import get_connection
from ..database.repository import (
    find_document_by_path,
    find_document_by_hash,
    delete_document_by_path,
    update_document_path
)


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md"
}

DEBOUNCE_SECONDS = 1.0


class DocumentEventHandler(FileSystemEventHandler):

    def __init__(self):
        super().__init__()
        self.last_event_times = {}

    def _is_supported_file(self, file_path):
        return (
            Path(file_path).suffix.lower()
            in SUPPORTED_EXTENSIONS
        )

    def _is_file_ready(self, file_path, retries=5, delay=0.5):
        path = Path(file_path)

        for _ in range(retries):
            if not path.exists() or not path.is_file():
                return False

            try:
                if path.stat().st_size == 0:
                    time.sleep(delay)
                    continue

                with open(path, "rb") as file:
                    file.read(1)

                return True

            except (PermissionError, OSError):
                time.sleep(delay)

        return False

    def _should_process(self, file_path):
        now = time.time()
        key = str(Path(file_path).resolve())

        last_event = self.last_event_times.get(key)

        if (
            last_event is not None
            and now - last_event < DEBOUNCE_SECONDS
        ):
            return False

        self.last_event_times[key] = now
        return True

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

    def index_file(self, file_path):
        file_path = Path(file_path)

        if not self._is_file_ready(file_path):
            print(
                f"Skipped {file_path.name}: "
                "file is empty or unavailable."
            )
            return

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

    def sync_modified_file(self, file_path):
        file_path = Path(file_path)

        if not self._is_file_ready(file_path):
            return

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

                self.index_file(file_path)
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

            self.index_file(file_path)

        except Exception as error:
            print(
                f"Failed to process modified file "
                f"{file_path.name}: {error}"
            )

    def on_created(self, event):
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        if not self._is_supported_file(file_path):
            return

        if not self._should_process(file_path):
            return

        print(
            f"\nNew document detected: "
            f"{file_path.name}"
        )

        self.index_file(file_path)

    def on_modified(self, event):
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        if not self._is_supported_file(file_path):
            return

        if not self._should_process(file_path):
            return

        print(
            f"\nDocument modified: "
            f"{file_path.name}"
        )

        self.sync_modified_file(file_path)

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
                f"Removed "
                f"{deleted_document['filename']} "
                f"from Morrow "
                f"(Document ID: "
                f"{deleted_document['id']})"
            )

        except Exception as error:
            print(
                f"Failed to remove deleted document "
                f"{file_path.name}: {error}"
            )


def initial_scan(folder_path, event_handler):
    folder = Path(folder_path)

    print("\nPerforming initial folder scan...")

    files_found = 0
    files_indexed = 0
    paths_updated = 0

    for file_path in folder.iterdir():

        if not file_path.is_file():
            continue

        if not event_handler._is_supported_file(file_path):
            continue

        files_found += 1

        resolved_path = str(file_path.resolve())

        # First check whether this exact path is already known.
        existing_document = find_document_by_path(
            resolved_path
        )

        if existing_document is not None:
            print(
                f"Already indexed: "
                f"{file_path.name}"
            )
            continue

        # If the path is unknown, check whether the
        # same file content was previously indexed.
        try:
            file_hash = calculate_file_hash(
                str(file_path)
            )

            existing_by_hash = find_document_by_hash(
                file_hash
            )

            if existing_by_hash is not None:
                update_document_path(
                    document_id=existing_by_hash["id"],
                    file_path=resolved_path,
                    filename=file_path.name
                )

                print(
                    f"Updated existing record: "
                    f"{file_path.name} "
                    f"(Document ID: "
                    f"{existing_by_hash['id']})"
                )

                paths_updated += 1
                continue

        except Exception as error:
            print(
                f"Failed to check existing hash "
                f"for {file_path.name}: {error}"
            )
            continue

        # No path match and no hash match:
        # this is genuinely a new document.
        print(
            f"Found unindexed document: "
            f"{file_path.name}"
        )

        event_handler.index_file(file_path)
        files_indexed += 1

    print(
        f"Initial scan complete: "
        f"{files_found} supported files found, "
        f"{files_indexed} indexed, "
        f"{paths_updated} existing paths updated."
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

    # First synchronize files that already exist.
    initial_scan(
        folder,
        event_handler
    )

    observer = Observer()

    observer.schedule(
        event_handler,
        str(folder),
        recursive=False
    )

    observer.start()

    print("\n" + "=" * 60)
    print("MORROW DOCUMENT WATCHER")
    print("=" * 60)
    print(f"Watching: {folder.resolve()}")
    print(
        "Supported: "
        ".pdf, .docx, .txt, .md"
    )
    print(
        f"Debounce: "
        f"{DEBOUNCE_SECONDS} seconds"
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