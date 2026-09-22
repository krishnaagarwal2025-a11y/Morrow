def build_context(results):
    context_parts = []

    for index, result in enumerate(results, start=1):
        source = (
            f"Page {result['page_number']}"
            if result["page_number"] is not None
            else "Page unknown"
        )

        context_parts.append(
            f"""[Source {index}]
{source}
Chunk ID: {result['chunk_id']}

{result['text']}
"""
        )

    return "\n".join(context_parts)