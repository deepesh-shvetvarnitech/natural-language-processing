import numpy as np

from src.search import embedding_similar_reviews


class DummyModel:
    def encode(
        self,
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    ):
        return np.array([[1.0, 0.0]])


def test_similar_reviews():
    reviews = [
        "Battery backup is poor.",
        "The camera is excellent."
    ]

    embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    results = embedding_similar_reviews(
        "My phone battery drains fast.",
        reviews,
        DummyModel(),
        embeddings,
        top_k=1
    )

    assert len(results) == 1
    assert results[0]["review"] == "Battery backup is poor."
    assert results[0]["rank"] == 1