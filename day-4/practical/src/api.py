
from pathlib import Path
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.search import search_tfidf, search_semantic

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

app = FastAPI(title="Day 4 — Similar Review Search", version="1.0.0")

class SearchRequest(BaseModel):
    query: str = Field(min_length=1, description="Text to find similar reviews for")
    top_k: int = Field(default=5, ge=1, le=20)
    search_type: str = Field(default="semantic", pattern="^(semantic|tfidf)$")
    exclude_review_id: str | None = None

@lru_cache(maxsize=1)
def load_artifacts():
    required = [
        DATA_DIR / "clean_reviews.csv",
        ARTIFACTS_DIR / "tfidf_vectorizer.joblib",
        ARTIFACTS_DIR / "tfidf_matrix.npz",
        ARTIFACTS_DIR / "review_embeddings.npy",
        ARTIFACTS_DIR / "embedding_model_name.txt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Artifacts missing. Run `python -m src.make_dataset` then `python -m src.main`.")
    df = pd.read_csv(required[0])
    vectorizer = joblib.load(required[1])
    tfidf_matrix = load_npz(required[2])
    embeddings = np.load(required[3])
    model_name = required[4].read_text(encoding="utf-8").strip()
    return df, vectorizer, tfidf_matrix, embeddings, model_name

@app.get("/health")
def health():
    return {"status": "ok", "service": "similar-review-search"}

@app.post("/similar-reviews")
def similar_reviews(request: SearchRequest):
    try:
        df, vectorizer, tfidf_matrix, embeddings, model_name = load_artifacts()
        if request.search_type == "tfidf":
            results = search_tfidf(request.query, df, vectorizer, tfidf_matrix,
                                   request.top_k, request.exclude_review_id)
        else:
            results = search_semantic(request.query, df, embeddings, request.top_k,
                                      request.exclude_review_id, model_name)
        return {"query": request.query, "search_type": request.search_type, "results": results}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
