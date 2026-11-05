
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from src.embeddings import MODEL_NAME, generate_embeddings
from src.preprocess import clean_text
from src.search import build_tfidf

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
OUTPUTS_DIR = BASE_DIR / "outputs"

def main():
    input_path = DATA_DIR / "product_reviews.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found. Run: python -m src.make_dataset")
    df = pd.read_csv(input_path)
    required = {"review_id", "review", "category"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {sorted(required)}")
    df["clean_review"] = df["review"].fillna("").map(clean_text)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATA_DIR / "clean_reviews.csv", index=False)

    texts = df["clean_review"].tolist()
    vectorizer, tfidf_matrix = build_tfidf(texts)
    embeddings = generate_embeddings(df["review"].fillna("").astype(str).tolist())

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    save_npz(ARTIFACTS_DIR / "tfidf_matrix.npz", tfidf_matrix)
    joblib.dump(vectorizer, ARTIFACTS_DIR / "tfidf_vectorizer.joblib")
    np.save(ARTIFACTS_DIR / "review_embeddings.npy", embeddings)
    (ARTIFACTS_DIR / "embedding_model_name.txt").write_text(MODEL_NAME, encoding="utf-8")
    vocab = vectorizer.get_feature_names_out().tolist()
    (OUTPUTS_DIR / "vocabulary.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    print(f"Built artifacts for {len(df)} reviews.")
    print(f"Embedding shape: {embeddings.shape}")

if __name__ == "__main__":
    main()
