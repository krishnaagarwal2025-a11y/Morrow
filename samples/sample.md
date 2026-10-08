# Morrow Knowledge Base

## Project Overview

Morrow is a local-first knowledge retrieval system designed to help users search and understand their personal documents.

The system is being developed as a modular pipeline. Each stage has a specific responsibility, starting with document ingestion and eventually leading to semantic retrieval and question answering.

## Document Ingestion

Morrow supports several document formats.

Currently supported formats include:

- PDF
- DOCX
- TXT
- Markdown

Each format is processed using an appropriate extractor.

PDF documents are processed using PyMuPDF. DOCX documents are processed using a DOCX-specific extractor, while TXT and Markdown files are processed as text documents.

## Chunking

After text extraction, documents are divided into smaller chunks.

Chunking allows the retrieval system to work with specific sections of a document instead of treating an entire document as one large piece of information.

Morrow currently uses configurable word-based chunking.

The default experimental configuration is:

- Chunk size: 500 words
- Overlap: 100 words

The overlap allows important information near the boundary of two chunks to appear in both chunks.

## Metadata

Every generated chunk can contain metadata.

Important metadata fields include:

- chunk_id
- chunk_index
- source
- file_type
- page_number
- word_count

Metadata will later be useful for filtering search results and showing citations to users.

## Embeddings

The next stage of Morrow is semantic embeddings.

An embedding converts text into a numerical vector representation.

For example, two sentences with similar meanings should produce vectors that are closer together than sentences discussing unrelated subjects.

The embedding stage will allow Morrow to perform semantic search.

## Vector Database

Morrow will use PostgreSQL with the pgvector extension to store embeddings.

PostgreSQL will store the document metadata and chunk information, while pgvector will allow numerical vectors to be indexed and compared.

## Future Features

Potential future improvements include:

1. Semantic search
2. Hybrid keyword and vector search
3. Reranking
4. Document versioning
5. Knowledge graphs
6. Local language models
7. OCR for scanned documents
8. Automatic folder monitoring
9. Search evaluation
10. Source-aware question answering

## Design Philosophy

Morrow is designed around three principles:

### Privacy

User documents should remain local whenever possible.

### Modularity

Each processing stage should be independent and replaceable.

### Explainability

Retrieved information should retain enough metadata to identify its original source.