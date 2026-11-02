
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocess import (
    clean_text,
    tokenize,
    remove_stopwords,
    stem_tokens,
    lemmatize_tokens,
    get_pos_tags,
    preprocess,
)

app = FastAPI(
    title="Day 2 NLP Text Features API",
    description="Assignment endpoints for text preprocessing and TF-IDF",
    version="1.0.0",
)


class TextRequest(BaseModel):
    text: str = Field(min_length=1)


class TextsRequest(BaseModel):
    texts: list[str] = Field(min_length=2)


class StemLemmatizeRequest(BaseModel):
    tokens: list[str] = Field(min_length=1)


def get_top_terms(matrix, feature_names, top_n=10):
    average_scores = np.asarray(matrix.mean(axis=0)).ravel()
    top_indices = average_scores.argsort()[::-1][:top_n]

    return [
        {
            "feature": feature_names[index],
            "average_tfidf": round(float(average_scores[index]), 6),
        }
        for index in top_indices
    ]


def build_ngram_report(texts, ngram_range):
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=5000,
        min_df=1,
        max_df=1.0,
    )

    matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out().tolist()

    total_values = matrix.shape[0] * matrix.shape[1]
    nonzero_values = matrix.nnz

    sparsity = (
        100 * (1 - nonzero_values / total_values)
        if total_values
        else 0.0
    )

    return {
        "vocabulary_size": len(feature_names),
        "feature_count": len(feature_names),
        "first_20_features": feature_names[:20],
        "top_10_features": get_top_terms(matrix, feature_names, 10),
        "matrix_shape": list(matrix.shape),
        "sparsity_percentage": round(float(sparsity), 2),
    }


@app.get("/")
def home():
    return {
        "message": "Day 2 NLP API is running",
        "docs": "/docs",
        "endpoints": [
            "/clean",
            "/tokenize",
            "/stemming",
            "/pos-tagging",
            "/preprocess",
            "/tfidf-conversion",
            "/features",
            "/feature-report",
        ],
    }


@app.post("/clean")
def clean_endpoint(request: TextRequest):
    return {
        "original_text": request.text,
        "cleaned_text": clean_text(request.text),
    }


@app.post("/tokenize")
def tokenize_endpoint(request: TextRequest):
    cleaned_text = clean_text(request.text)
    original_tokens = tokenize(cleaned_text)
    filtered_tokens = remove_stopwords(original_tokens)

    return {
        "original_text": request.text,
        "original_tokens": original_tokens,
        "tokens_after_stopword_removal": filtered_tokens,
    }


@app.post("/stemming")
def stemming_endpoint(request: StemLemmatizeRequest):
    original_tokens = request.tokens
    stemmed_tokens = stem_tokens(original_tokens)
    lemmatized_tokens = lemmatize_tokens(original_tokens)

    return {
        "original_tokens": original_tokens,
        "stemmed_tokens": stemmed_tokens,
        "lemmatized_tokens": lemmatized_tokens,
        "comparison": (
            "Stemming cuts word endings and may create word fragments. "
            "Lemmatization uses linguistic analysis to return base forms."
        ),
    }


@app.post("/pos-tagging")
def pos_tagging_endpoint(request: TextRequest):
    return {
        "text": request.text,
        "pos_tags": get_pos_tags(request.text),
    }


@app.post("/preprocess")
def preprocess_endpoint(request: TextRequest):
    original_text = request.text

    cleaned_text = clean_text(original_text)
    original_tokens = tokenize(cleaned_text)
    filtered_tokens = remove_stopwords(original_tokens)
    lemmas = lemmatize_tokens(filtered_tokens)

    processed_text = preprocess(
        original_text,
        lowercase=True,
        remove_html=True,
        remove_urls=True,
        remove_stopwords_flag=True,
        use_stemming=False,
        use_lemmatization=True,
    )

    return {
        "original_text": original_text,
        "cleaned_text": cleaned_text,
        "original_tokens": original_tokens,
        "filtered_tokens": filtered_tokens,
        "lemmas": lemmas,
        "processed_text": processed_text,
        "token_count": len(original_tokens),
    }


@app.post("/tfidf-conversion")
def tfidf_conversion_endpoint(request: TextsRequest):
    try:
        vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 1),
        )

        matrix = vectorizer.fit_transform(request.texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        return {
            "matrix_shape": list(matrix.shape),
            "feature_names": feature_names,
            "feature_count": len(feature_names),
            "tfidf_matrix": matrix.toarray().tolist(),
            "matrix_format": (
                "Sparse TF-IDF matrix converted to dense list for Swagger"
            ),
            "vectorizer_parameters": {
                "max_features": 5000,
                "min_df": 2,
                "max_df": 0.95,
                "ngram_range": [1, 1],
            },
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "TF-IDF could not be created. Check the number of reviews "
                "and the min_df/max_df settings. "
                f"Details: {str(exc)}"
            ),
        )


@app.post("/features")
def features_endpoint(request: TextsRequest):
    processed_texts = [preprocess(text) for text in request.texts]

    try:
        unigram = build_ngram_report(processed_texts, (1, 1))
        bigram = build_ngram_report(processed_texts, (2, 2))
        combined = build_ngram_report(processed_texts, (1, 2))

        return {
            "processed_texts": processed_texts,
            "unigram": unigram,
            "bigram": bigram,
            "unigram_and_bigram": combined,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to build n-gram features: {str(exc)}",
        )


@app.post("/feature-report")
def feature_report_endpoint(request: TextsRequest):
    processed_texts = [preprocess(text) for text in request.texts]

    try:
        return {
            "processed_texts": processed_texts,
            "unigram_only": build_ngram_report(processed_texts, (1, 1)),
            "bigram_only": build_ngram_report(processed_texts, (2, 2)),
            "unigram_and_bigram": build_ngram_report(processed_texts, (1, 2)),
            "answers": {
                "phrases": (
                    "Bigram-only and unigram+bigram representations capture "
                    "phrases such as battery life and camera quality."
                ),
                "memory": (
                    "Many bigrams increase the vocabulary size and the number "
                    "of matrix entries that must be stored or processed."
                ),
                "min_df_max_df": (
                    "min_df removes terms appearing in too few documents. "
                    "max_df removes terms appearing in too many documents."
                ),
                "inference": (
                    "Reuse the same fitted vectorizer so the vocabulary and "
                    "feature order remain consistent."
                ),
                "fastapi": (
                    "Expose preprocessing through a POST endpoint that accepts "
                    "text, applies the fitted preprocessing logic, and returns "
                    "processed text or TF-IDF features."
                ),
            },
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to build feature report: {str(exc)}",
        )
