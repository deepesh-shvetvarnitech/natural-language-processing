import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = MODEL_NAME):
    return SentenceTransformer(model_name)


def create_embeddings(texts: list[str], model):
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )


def embed_query(query: str, model):
    return model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )