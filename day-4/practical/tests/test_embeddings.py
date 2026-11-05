import numpy as np
from src.embeddings import cosine_similarity

def test_cosine_similarity():
    query = np.array([1.0, 0.0])
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    scores = cosine_similarity(query, matrix)
    assert np.allclose(scores, [1.0, 0.0, -1.0])
