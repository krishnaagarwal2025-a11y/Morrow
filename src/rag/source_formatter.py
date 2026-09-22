def format_sources(results):
    sources = []

    for result in results:
        sources.append({
            "filename": result["filename"],
            "page_number": result["page_number"],
            "chunk_id": result["chunk_id"],
            "similarity": result["similarity"]
        })

    return sources