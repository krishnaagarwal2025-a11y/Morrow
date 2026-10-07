def format_sources(results):
    sources = []

    for result in results:
        sources.append({
            "filename": result["filename"],
            "page_number": result["page_number"],
            "chunk_id": result["chunk_id"],
            "rrf_score": result.get("rrf_score"),
            "vector_rank": result.get("vector_rank"),
            "keyword_rank": result.get("keyword_rank")
        })

    return sources