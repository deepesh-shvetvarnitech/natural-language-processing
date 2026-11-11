import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import spacy

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from src.schemas import (
    ReviewRequest,
    KeywordRequest,
    SimilarRequest,
    AnalyzeRequest
)
from src.extraction import (
    load_domain_terms,
    extract_entities,
    extract_custom_entities,
    extract_keywords,
    extract_issues
)
from src.search import embedding_similar_reviews
from src.sentiment import predict_sentiment
from src.embeddings import load_embedding_model


BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("review_intelligence")


from src.schemas import (
    ReviewRequest,
    KeywordRequest,
    SimilarRequest,
    AnalyzeRequest
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.sentiment_model = joblib.load(
        ARTIFACT_DIR / "sentiment_model.joblib"
    )

    app.state.vectorizer = joblib.load(
        ARTIFACT_DIR / "tfidf_vectorizer.joblib"
    )

    app.state.nlp = spacy.load("en_core_web_sm")

    app.state.domain_terms = load_domain_terms(
        str(DATA_DIR / "domain_terms.json")
    )

    app.state.embedding_model = load_embedding_model()

    app.state.review_embeddings = np.load(
        ARTIFACT_DIR / "review_embeddings.npy"
    )

    reviews_df = pd.read_csv(ARTIFACT_DIR / "reviews.csv")
    app.state.reviews = reviews_df["review"].astype(str).tolist()

    yield


app = FastAPI(
    title="Day 5 Review Intelligence Service",
    version="1.0.0",
    lifespan=lifespan
)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
        latency = round((time.perf_counter() - start_time) * 1000, 2)

        response.headers["X-Request-ID"] = request_id

        logger.info(
            "request_id=%s method=%s path=%s status=%s latency_ms=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            latency
        )

        return response

    except Exception:
        logger.exception("request_id=%s request failed", request_id)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal inference error"},
            headers={"X-Request-ID": request_id}
        )


def validate_review(review: str):
    if not review or not review.strip():
        raise HTTPException(
            status_code=400,
            detail="Review must not be empty"
        )

    if len(review) > 5000:
        raise HTTPException(
            status_code=400,
            detail="Review must be 5000 characters or fewer"
        )


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/sentiment")
def sentiment(request: ReviewRequest):
    validate_review(request.review)

    try:
        return predict_sentiment(
            request.review,
            app.state.sentiment_model,
            app.state.vectorizer
        )
    except Exception:
        logger.exception("Sentiment inference failed")
        raise HTTPException(
            status_code=500,
            detail="Sentiment inference failed"
        )


@app.post("/keywords")
def keywords(request: KeywordRequest):
    validate_review(request.review)

    try:
        return {
            "keywords": extract_keywords(
                request.review,
                app.state.vectorizer,
                request.top_n
            )
        }
    except Exception:
        logger.exception("Keyword extraction failed")
        raise HTTPException(
            status_code=500,
            detail="Keyword extraction failed"
        )


@app.post("/entities")
def entities(request: ReviewRequest):
    validate_review(request.review)

    try:
        standard = extract_entities(
            request.review,
            app.state.nlp
        )

        custom = extract_custom_entities(
            request.review,
            app.state.nlp,
            app.state.domain_terms
        )

        issues = extract_issues(
            request.review,
            app.state.nlp,
            app.state.domain_terms
        )

        return {
            "entities": standard,
            "custom_entities": custom,
            "issues": issues
        }
    except Exception:
        logger.exception("Entity extraction failed")
        raise HTTPException(
            status_code=500,
            detail="Entity extraction failed"
        )


@app.post("/similar")
def similar(request: SimilarRequest):
    validate_review(request.review)

    try:
        results = embedding_similar_reviews(
            request.review,
            app.state.reviews,
            app.state.embedding_model,
            app.state.review_embeddings,
            request.top_k
        )

        return {"similar_reviews": results}
    except Exception:
        logger.exception("Semantic search failed")
        raise HTTPException(
            status_code=500,
            detail="Semantic search failed"
        )


@app.post("/analyze-review")
def analyze(request: AnalyzeRequest):
    validate_review(request.review)

    try:
        sentiment_result = predict_sentiment(
            request.review,
            app.state.sentiment_model,
            app.state.vectorizer
        )

        keywords_result = extract_keywords(
            request.review,
            app.state.vectorizer,
            request.top_keywords
        )

        standard_entities = extract_entities(
            request.review,
            app.state.nlp
        )

        custom_entities = extract_custom_entities(
            request.review,
            app.state.nlp,
            app.state.domain_terms
        )

        issues_result = extract_issues(
            request.review,
            app.state.nlp,
            app.state.domain_terms
        )

        similar_result = embedding_similar_reviews(
            request.review,
            app.state.reviews,
            app.state.embedding_model,
            app.state.review_embeddings,
            request.similar_reviews
        )

        return {
            "sentiment": sentiment_result["sentiment"],
            "confidence": sentiment_result["confidence"],
            "keywords": keywords_result,
            "entities": standard_entities + custom_entities,
            "issues": issues_result,
            "similar_reviews": similar_result
        }

    except Exception:
        logger.exception("Review analysis failed")
        raise HTTPException(
            status_code=500,
            detail="Review analysis failed"
        )