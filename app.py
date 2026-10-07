import streamlit as st
import tempfile
from pathlib import Path

from src.database.ingest_document import ingest_document
from src.database.repository import (
    get_all_documents,
    delete_document
)
from src.rag.rag_pipeline import RAGPipeline


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Morrow",
    page_icon="🌙"
)


# ==================================================
# CACHED RAG PIPELINE
# ==================================================

@st.cache_resource
def get_rag_pipeline():
    return RAGPipeline(
        top_k=5,
        min_similarity=0.2554
    )


# ==================================================
# HEADER
# ==================================================

st.title("🌙 Morrow")

st.write(
    "Local document question-answering using "
    "semantic retrieval and RAG."
)


# ==================================================
# DOCUMENT INDEXING
# ==================================================

st.header("📄 Index Documents")

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx", "txt", "md"]
)


if uploaded_file is not None:

    st.success(
        f"File selected: {uploaded_file.name}"
    )

    st.write(
        f"File size: {uploaded_file.size:,} bytes"
    )

    if st.button("Index Document"):

        try:

            # Preserve original file extension
            suffix = Path(
                uploaded_file.name
            ).suffix

            # Create temporary file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                temp_path = temp_file.name

            # Index document
            with st.spinner(
                "Indexing document..."
            ):

                document_id = ingest_document(
                    temp_path,
                    original_filename=uploaded_file.name
                )

            st.success(
                f"Document indexed successfully! "
                f"Document ID: {document_id}"
            )

            # Refresh page so document list updates
            st.rerun()

        except ValueError as error:

            # Duplicate document
            st.warning(
                f"⚠️ {error}"
            )

        except Exception as error:

            st.error(
                f"Failed to index document: {error}"
            )


# ==================================================
# INDEXED DOCUMENTS
# ==================================================

st.divider()

st.header("📚 Indexed Documents")

try:

    documents = get_all_documents()

    if not documents:

        st.info(
            "No documents have been indexed yet."
        )

    else:

        for document in documents:

            document_id = document["id"]
            filename = document["filename"]
            file_type = document["file_type"]
            created_at = document["created_at"]

            with st.container(border=True):

                col1, col2 = st.columns(
                    [4, 1]
                )

                with col1:

                    st.write(
                        f"📄 **{filename}**"
                    )

                    st.caption(
                        f"Type: {file_type}  |  "
                        f"Document ID: {document_id}  |  "
                        f"Indexed: {created_at}"
                    )

                with col2:

                    delete_key = (
                        f"delete_{document_id}"
                    )

                    if st.button(
                        "Delete",
                        key=delete_key
                    ):

                        st.session_state[
                            f"confirm_delete_{document_id}"
                        ] = True

                # ------------------------------------------
                # Delete confirmation
                # ------------------------------------------

                if st.session_state.get(
                    f"confirm_delete_{document_id}",
                    False
                ):

                    st.warning(
                        f"Are you sure you want to remove "
                        f"**{filename}** from Morrow?"
                    )

                    confirm_col1, confirm_col2 = st.columns(
                        2
                    )

                    with confirm_col1:

                        if st.button(
                            "Yes, delete",
                            key=f"confirm_{document_id}"
                        ):

                            try:

                                deleted_filename = (
                                    delete_document(
                                        document_id
                                    )
                                )

                                if deleted_filename:

                                    st.success(
                                        f"Removed "
                                        f"{deleted_filename} "
                                        f"from Morrow."
                                    )

                                else:

                                    st.warning(
                                        "Document was not found."
                                    )

                                st.session_state[
                                    f"confirm_delete_{document_id}"
                                ] = False

                                st.rerun()

                            except Exception as error:

                                st.error(
                                    f"Failed to delete "
                                    f"document: {error}"
                                )

                    with confirm_col2:

                        if st.button(
                            "Cancel",
                            key=f"cancel_{document_id}"
                        ):

                            st.session_state[
                                f"confirm_delete_{document_id}"
                            ] = False

                            st.rerun()

except Exception as error:

    st.error(
        f"Failed to load indexed documents: {error}"
    )


# ==================================================
# ASK MORROW
# ==================================================

st.divider()

st.header("💬 Ask Morrow")

question = st.text_input(
    "Ask a question about your indexed documents",
    placeholder="e.g. What database does Morrow use?"
)


if st.button("Ask Morrow"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        try:

            with st.spinner(
                "Searching documents and generating answer..."
            ):

                pipeline = get_rag_pipeline()

                result = pipeline.answer(
                    question
                )

            # ------------------------------------------
            # Answer
            # ------------------------------------------

            st.subheader("Answer")

            st.write(
                result["answer"]
            )

            # ------------------------------------------
            # Sources
            # ------------------------------------------

            st.subheader("Sources")

            if result["sources"]:

                for source in result["sources"]:

                    filename = source["filename"]
                    page = source["page_number"]

                    rrf_score = source.get("rrf_score")
                    vector_rank = source.get("vector_rank")
                    keyword_rank = source.get("keyword_rank")

                    if page is not None:

                        location = (
                            f"Page {page}"
                        )

                    else:

                        location = (
                            "Page unknown"
                        )

                    st.write(
                        f"📄 **{filename}** — "
                        f"{location} — "
                        f"Chunk: {source['chunk_id']}"
                    )

                    st.caption(
                        f"RRF Score: "
                        f"{rrf_score:.6f} | "
                        f"Vector Rank: "
                        f"{vector_rank if vector_rank is not None else 'Not retrieved'} | "
                        f"Keyword Rank: "
                        f"{keyword_rank if keyword_rank is not None else 'Not retrieved'}"
                    )

            else:

                st.write(
                    "No sources found."
                )

            # ------------------------------------------
            # Developer retrieved context
            # ------------------------------------------

            if "retrieved_context" in result:

                with st.expander(
                    "🔍 Developer: Retrieved Context"
                ):

                    st.text(
                        result["retrieved_context"]
                    )

        except Exception as error:

            st.error(
                f"Failed to answer question: {error}"
            )