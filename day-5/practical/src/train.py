from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from src.preprocess import preprocess_text
from src.embeddings import load_embedding_model, create_embeddings


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "product_reviews.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["review", "sentiment"])

    df["clean_review"] = df["review"].astype(str).apply(preprocess_text)

    X = df["clean_review"]
    y = df["sentiment"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vectors, y_train)

    predictions = model.predict(X_test_vectors)
    print(classification_report(y_test, predictions))

    joblib.dump(
        model,
        ARTIFACT_DIR / "sentiment_model.joblib"
    )

    joblib.dump(
        vectorizer,
        ARTIFACT_DIR / "tfidf_vectorizer.joblib"
    )

    reviews = df["review"].astype(str).tolist()

    embedding_model = load_embedding_model()
    review_embeddings = create_embeddings(
        reviews,
        embedding_model
    )

    np.save(
        ARTIFACT_DIR / "review_embeddings.npy",
        review_embeddings
    )

    with open(
        ARTIFACT_DIR / "embedding_model_name.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write("sentence-transformers/all-MiniLM-L6-v2")

    df[["review_id", "review", "sentiment"]].to_csv(
        ARTIFACT_DIR / "reviews.csv",
        index=False
    )


if __name__ == "__main__":
    main()