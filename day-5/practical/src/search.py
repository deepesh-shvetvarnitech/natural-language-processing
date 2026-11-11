import numpy as np
from src.embeddings import embed_query


def embedding_similar_reviews(
    query: str,
    reviews: list[str],
    model,
    review_embeddings,
    top_k: int = 3
) -> list[dict]:
    if not reviews or review_embeddings is None:
        return []

    query_embedding = embed_query(query, model)[0]
    scores = np.dot(review_embeddings, query_embedding)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, index in enumerate(top_indices, start=1):
        results.append({
            "review_id": int(index + 1),
            "review": reviews[index],
            "score": round(float(scores[index]), 4),
            "rank": rank
        })

    return results