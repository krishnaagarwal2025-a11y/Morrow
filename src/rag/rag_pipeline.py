from ..embeddings.embedder import Embedder
from ..database.retrieval import search_similar_chunks

from .context_builder import build_context
from .prompt import build_rag_prompt
from .llm import generate_response
from .source_formatter import format_sources


class RAGPipeline:

    def __init__(self, top_k=3, min_similarity=None):
        self.embedder = Embedder()
        self.top_k = top_k
        self.min_similarity = min_similarity

    def answer(self, question):

        # 1. Convert the question into an embedding
        query_embedding = self.embedder.embed_text(question)

        # 2. Retrieve relevant chunks from PostgreSQL + pgvector
        results = search_similar_chunks(
            query_embedding,
            top_k=self.top_k,
            min_similarity=self.min_similarity
        )

        # 3. If no sufficiently relevant chunks were found,
        #    do not send an empty/irrelevant context to the LLM.
        if not results:
            return {
                "question": question,
                "answer": (
                    "I could not find relevant information "
                    "in the provided documents."
                ),
                "sources": []
            }

        # 4. Build context from retrieved chunks
        context = build_context(results)

        # 5. Build the RAG prompt
        prompt = build_rag_prompt(
            question,
            context
        )


        # 6. Generate the answer using Gemma
        answer = generate_response(prompt)

        # 7. Format source information separately
        sources = format_sources(results)

        # 8. Return the complete RAG result
        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }