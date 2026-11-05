
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from src.search import search_tfidf, search_semantic

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"

def precision_recall_at_k(results, relevant_ids, k):
    retrieved = [item["review_id"] for item in results[:k]]
    hits = len(set(retrieved) & set(relevant_ids))
    precision = hits / k if k else 0.0
    recall = hits / len(relevant_ids) if relevant_ids else 0.0
    return precision, recall

def main():
    df = pd.read_csv(DATA_DIR / "clean_reviews.csv")
    vectorizer = joblib.load(ARTIFACTS_DIR / "tfidf_vectorizer.joblib")
    matrix = load_npz(ARTIFACTS_DIR / "tfidf_matrix.npz")
    embeddings = np.load(ARTIFACTS_DIR / "review_embeddings.npy")
    model_name = (ARTIFACTS_DIR / "embedding_model_name.txt").read_text(encoding="utf-8").strip()

    lines = ["# Search Evaluation", "", "Evaluation uses same-category reviews as relevant results. Each query review is excluded from its own result list.", ""]
    for method in ("tfidf", "semantic"):
        p_values, r_values, ranks = [], [], []
        for _, row in df.iterrows():
            relevant = set(df.loc[df["category"] == row["category"], "review_id"].astype(str))
            relevant.discard(str(row["review_id"]))
            if not relevant:
                continue
            if method == "tfidf":
                results = search_tfidf(row["review"], df, vectorizer, matrix, top_k=5,
                                       exclude_review_id=str(row["review_id"]))
            else:
                results = search_semantic(row["review"], df, embeddings, top_k=5,
                                          exclude_review_id=str(row["review_id"]), model_name=model_name)
            precision, recall = precision_recall_at_k(results, relevant, 5)
            p_values.append(precision); r_values.append(recall)
            first_rank = next((i for i, item in enumerate(results, 1) if item["review_id"] in relevant), None)
            ranks.append(first_rank if first_rank is not None else 6)
        lines += [f"## {method.upper()}", f"- Mean Precision@5: {np.mean(p_values):.3f}",
                  f"- Mean Recall@5: {np.mean(r_values):.3f}",
                  f"- Mean first relevant rank (missing counted as 6): {np.mean(ranks):.3f}", ""]
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "evaluation.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
