from ..embeddings.embedder import Embedder
from src.database.retrieval import search_hybrid_rrf

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

        # 1. Retrieve relevant chunks using RRF hybrid retrieval
        results = search_hybrid_rrf(
            query=question,
            embedder=self.embedder,
            top_k=self.top_k,
            candidate_k=5
        )

        # 2. If no relevant chunks were found,
        #    do not send empty/irrelevant context to the LLM.
        if not results:
            return {
                "question": question,
                "answer": (
                    "I could not find relevant information "
                    "in the provided documents."
                ),
                "sources": []
            }

        # 3. Build context from retrieved chunks
        context = build_context(results)

        # 4. Build the RAG prompt
        prompt = build_rag_prompt(
            question,
            context
        )

        # 5. Generate the answer using Gemma
        answer = generate_response(prompt)

        # 6. Format source information separately
        sources = format_sources(results)

        # 7. Return the complete RAG result
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_context": context
        }