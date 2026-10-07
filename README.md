# Morrow

> A local-first, document-grounded AI system for automatically indexing documents and answering natural-language questions using semantic and hybrid retrieval.

Morrow is a learning-focused RAG (Retrieval-Augmented Generation) system designed to turn a collection of local documents into searchable knowledge.

Instead of manually organizing information into a knowledge base, Morrow can ingest documents, extract their content, split them into chunks, generate vector embeddings, store them in PostgreSQL with pgvector, retrieve relevant information using hybrid search, and pass the retrieved context to a local language model to generate grounded answers.

The project is being developed primarily as an engineering and learning project, with an emphasis on understanding how document ingestion, embeddings, vector databases, information retrieval, RAG pipelines, local LLMs, and evaluation systems work together.

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Core Architecture](#core-architecture)
- [How Morrow Works](#how-morrow-works)
- [Current Features](#current-features)
- [Supported Documents](#supported-documents)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Database Design](#database-design)
- [Retrieval System](#retrieval-system)
- [RAG Pipeline](#rag-pipeline)
- [Automatic Folder Synchronization](#automatic-folder-synchronization)
- [Evaluation](#evaluation)
- [Running Morrow](#running-morrow)
- [Environment Configuration](#environment-configuration)
- [Development Milestones](#development-milestones)
- [Current Limitations](#current-limitations)
- [Roadmap](#roadmap)
- [Learning Goals](#learning-goals)
- [License](#license)

---

# Overview

Morrow is a local-first document question-answering system.

The system accepts documents such as:

- PDF
- DOCX
- TXT
- Markdown

and processes them through an ingestion and retrieval pipeline.

At a high level:

```text
Local Documents
       │
       ▼
Document Extraction
       │
       ▼
Chunking
       │
       ▼
Embedding Generation
       │
       ▼
PostgreSQL + pgvector
       │
       ▼
Hybrid Retrieval
(Vector + Keyword + RRF)
       │
       ▼
Relevant Context
       │
       ▼
Local LLM
       │
       ▼
Grounded Answer + Sources