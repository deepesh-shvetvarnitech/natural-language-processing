import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "review_id": ["A", "B", "C"],
        "review": ["battery lasts long", "camera is sharp", "battery drains quickly"],
        "clean_review": ["battery lasts long", "camera is sharp", "battery drains quickly"],
        "category": ["electronics", "electronics", "electronics"],
    })

@pytest.fixture
def sample_tfidf(sample_df):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(sample_df["clean_review"])
    return vectorizer, matrix
