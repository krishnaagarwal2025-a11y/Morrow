import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.repository import insert_document

document_id = insert_document(
    filename="repository_test.pdf",
    file_path="C:\\projects\\Morrow\\repository_test.pdf",
    file_type=".pdf"
)


print("Inserted document ID:", document_id)