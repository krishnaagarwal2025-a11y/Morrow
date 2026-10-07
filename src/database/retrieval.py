from .connection import get_connection


def search_similar_chunks(
    query_embedding,
    top_k=5,
    min_similarity=None
):
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    connection = get_connection()

    try:
        cursor = connection.cursor()

        if min_similarity is None:

            cursor.execute(
                """
                SELECT
                    chunks.id,
                    chunks.document_id,
                    documents.filename,
                    chunks.chunk_index,
                    chunks.text,
                    chunks.page_number,
                    chunks.word_count,
                    1 - (chunks.embedding <=> %s) AS similarity
                FROM chunks
                JOIN documents
                    ON chunks.document_id = documents.id
                WHERE chunks.embedding IS NOT NULL
                ORDER BY chunks.embedding <=> %s
                LIMIT %s;
                """,
                (
                    query_embedding,
                    query_embedding,
                    top_k
                )
            )

        else:

            cursor.execute(
                """
                SELECT
                    chunks.id,
                    chunks.document_id,
                    documents.filename,
                    chunks.chunk_index,
                    chunks.text,
                    chunks.page_number,
                    chunks.word_count,
                    1 - (chunks.embedding <=> %s) AS similarity
                FROM chunks
                JOIN documents
                    ON chunks.document_id = documents.id
                WHERE
                    chunks.embedding IS NOT NULL
                    AND 1 - (chunks.embedding <=> %s) >= %s
                ORDER BY chunks.embedding <=> %s
                LIMIT %s;
                """,
                (
                    query_embedding,
                    query_embedding,
                    min_similarity,
                    query_embedding,
                    top_k
                )
            )

        rows = cursor.fetchall()

        results = []

        for row in rows:
            results.append({
                "chunk_id": row[0],
                "document_id": row[1],
                "filename": row[2],
                "chunk_index": row[3],
                "text": row[4],
                "page_number": row[5],
                "word_count": row[6],
                "similarity": float(row[7])
            })

        return results

    finally:
        connection.close()

def search_keyword_chunks(
    query,
    top_k=5
):
    if not query or not query.strip():
        raise ValueError("query must not be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                chunks.id,
                chunks.document_id,
                documents.filename,
                chunks.chunk_index,
                chunks.text,
                chunks.page_number,
                chunks.word_count,
                ts_rank(
                    to_tsvector('english', chunks.text),
                    plainto_tsquery('english', %s)
                ) AS keyword_score
            FROM chunks
            JOIN documents
                ON chunks.document_id = documents.id
            WHERE to_tsvector(
                'english',
                chunks.text
            ) @@ plainto_tsquery(
                'english',
                %s
            )
            ORDER BY keyword_score DESC
            LIMIT %s;
            """,
            (
                query,
                query,
                top_k
            )
        )

        rows = cursor.fetchall()

        results = []

        for row in rows:
            results.append({
                "chunk_id": row[0],
                "document_id": row[1],
                "filename": row[2],
                "chunk_index": row[3],
                "text": row[4],
                "page_number": row[5],
                "word_count": row[6],
                "keyword_score": float(row[7])
            })

        return results

    finally:
        connection.close()


def search_hybrid_chunks(
    query,
    embedder,
    top_k=5,
    vector_weight=0.7,
    keyword_weight=0.3
):
    if not query or not query.strip():
        raise ValueError("query must not be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if vector_weight < 0 or keyword_weight < 0:
        raise ValueError(
            "weights must not be negative"
        )

    total_weight = vector_weight + keyword_weight

    if total_weight == 0:
        raise ValueError(
            "at least one weight must be greater than 0"
        )

    vector_weight /= total_weight
    keyword_weight /= total_weight

    query_embedding = embedder.embed_text(query)

    vector_results = search_similar_chunks(
        query_embedding,
        top_k=top_k
    )

    keyword_results = search_keyword_chunks(
        query,
        top_k=top_k
    )

    normalize_scores(
        vector_results,
        "similarity"
    )

    normalize_scores(
        keyword_results,
        "keyword_score"
    )

    candidates = {}

    for result in vector_results:
        chunk_id = result["chunk_id"]

        candidates[chunk_id] = {
            **result,
            "vector_score": result["similarity"],
            "keyword_score": 0.0,
            "normalized_vector_score":
                result["normalized_similarity"],
            "normalized_keyword_score": 0.0
        }

    for result in keyword_results:
        chunk_id = result["chunk_id"]

        if chunk_id not in candidates:
            candidates[chunk_id] = {
                **result,
                "vector_score": 0.0,
                "keyword_score":
                    result["keyword_score"],
                "normalized_vector_score": 0.0,
                "normalized_keyword_score":
                    result["normalized_keyword_score"]
            }
        else:
            candidates[chunk_id][
                "keyword_score"
            ] = result["keyword_score"]

            candidates[chunk_id][
                "normalized_keyword_score"
            ] = result[
                "normalized_keyword_score"
            ]

    for result in candidates.values():
        result["hybrid_score"] = (
            vector_weight
            * result["normalized_vector_score"]
            +
            keyword_weight
            * result["normalized_keyword_score"]
        )

    ranked_results = sorted(
        candidates.values(),
        key=lambda result: result["hybrid_score"],
        reverse=True
    )

    return ranked_results[:top_k]

def search_hybrid_rrf(
    query,
    embedder,
    top_k=5,
    candidate_k=10,
    rrf_k=60
):
    """
    Hybrid retrieval using Reciprocal Rank Fusion (RRF).

    Combines vector-search and keyword-search rankings
    without comparing their raw score scales.
    """

    if not query or not query.strip():
        raise ValueError("query must not be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if candidate_k <= 0:
        raise ValueError("candidate_k must be greater than 0")

    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0")

    # Generate query embedding once
    query_embedding = embedder.embed_text(query)

    # Retrieve candidates from both systems
    vector_results = search_similar_chunks(
        query_embedding,
        top_k=candidate_k
    )

    keyword_results = search_keyword_chunks(
        query,
        top_k=candidate_k
    )

    candidates = {}

    # ---------------------------------------------------------
    # Vector ranking
    # ---------------------------------------------------------
    for rank, result in enumerate(vector_results, start=1):
        chunk_id = result["chunk_id"]

        candidates[chunk_id] = {
            **result,
            "vector_rank": rank,
            "keyword_rank": None,
            "rrf_score": 1 / (rrf_k + rank),
            "vector_score": result.get("similarity", 0.0),
            "keyword_score": 0.0,
        }

    # ---------------------------------------------------------
    # Keyword ranking
    # ---------------------------------------------------------
    for rank, result in enumerate(keyword_results, start=1):
        chunk_id = result["chunk_id"]

        if chunk_id not in candidates:
            candidates[chunk_id] = {
                **result,
                "vector_rank": None,
                "keyword_rank": rank,
                "rrf_score": 1 / (rrf_k + rank),
                "vector_score": 0.0,
                "keyword_score": result.get("keyword_score", 0.0),
            }

        else:
            candidates[chunk_id]["keyword_rank"] = rank
            candidates[chunk_id]["keyword_score"] = result.get(
                "keyword_score",
                0.0
            )

            candidates[chunk_id]["rrf_score"] += (
                1 / (rrf_k + rank)
            )

    # ---------------------------------------------------------
    # Final ranking
    # ---------------------------------------------------------
    ranked_results = sorted(
        candidates.values(),
        key=lambda result: result["rrf_score"],
        reverse=True
    )

    return ranked_results[:top_k]

def normalize_scores(results, score_key):
    if not results:
        return results

    scores = [
        result[score_key]
        for result in results
    ]

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        for result in results:
            result[f"normalized_{score_key}"] = 0.0

        return results

    for result in results:
        result[f"normalized_{score_key}"] = (
            result[score_key] - min_score
        ) / (
            max_score - min_score
        )

    return results