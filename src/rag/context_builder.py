def build_context(results):
    context_parts = []

    for index, result in enumerate(results, start=1):
        if result["page_number"] is not None:
            source = f"Page {result['page_number']}"
        else:
            source = "Page unknown"

        context_parts.append(
            f"""
SOURCE {index}
Filename: {result['filename']}
Location: {source}
Chunk ID: {result['chunk_id']}

CONTENT:
{result['text']}

END SOURCE {index}
""".strip()
        )

    return "\n\n".join(context_parts)