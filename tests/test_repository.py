from .repository import insert_document


document_id = insert_document(
    filename="repository_test.pdf",
    file_path="C:\\projects\\Morrow\\repository_test.pdf",
    file_type=".pdf"
)


print("Inserted document ID:", document_id)