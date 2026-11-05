
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from src.embeddings import generate_embeddings
from src.preprocess import clean_text

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

def build_tfidf(texts: list[str]):
 
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix

def search_tfidf(query: str, df: pd.DataFrame, vectorizer, matrix,
                 top_k: int = 5, exclude_review_id: str | None = None) -> list[dict]:
   
    query_vector = vectorizer.transform([clean_text(query)])
    scores = sklearn_cosine(query_vector, matrix).ravel()
    return _rank_results(scores, df, top_k, exclude_review_id, "tfidf")

def search_semantic(query: str, df: pd.DataFrame, embeddings: np.ndarray,
                    top_k: int = 5, exclude_review_id: str | None = None,
                    model_name: str | None = None) -> list[dict]:
    
    query_vector = generate_embeddings([query], model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2")[0]
    scores = np.asarray(embeddings) @ query_vector
    return _rank_results(scores, df, top_k, exclude_review_id, "semantic")

def _rank_results(scores, df, top_k, exclude_review_id, search_type):
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    ranked = np.argsort(np.asarray(scores))[::-1]
    results = []
    for index in ranked:
        row = df.iloc[int(index)]
        review_id = str(row["review_id"])
        if exclude_review_id and review_id == exclude_review_id:
            continue
        results.append({
            "review_id": review_id,
            "review": str(row["review"]),
            "category": str(row["category"]),
            "score": round(float(scores[index]), 4),
            "search_type": search_type,
        })
        if len(results) >= top_k:
            break
    return results
