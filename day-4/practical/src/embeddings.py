
from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def load_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
   
    return SentenceTransformer(model_name)

def generate_embeddings(texts: list[str], model_name: str = MODEL_NAME) -> np.ndarray:
   
    model = load_embedding_model(model_name)
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)

def cosine_similarity(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
   
    query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    corpus = np.asarray(matrix, dtype=np.float32)
    query_norm = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    corpus_norm = corpus / np.maximum(np.linalg.norm(corpus, axis=1, keepdims=True), 1e-12)
    return (query_norm @ corpus_norm.T).ravel()
