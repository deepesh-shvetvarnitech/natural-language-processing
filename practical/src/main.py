<<<<<<< HEAD

import json
from pathlib import Path

import pandas as pd
from scipy.sparse import save_npz

from src.pipeline import process_reviews
from src.feature_extraction import (
    build_tfidf_vectorizer,
    get_top_tfidf_features,
    compare_ngram_features
)



BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "reviews.csv"
OUTPUT_DIR = BASE_DIR / "outputs"


def main():

   
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    
    df = pd.read_csv(DATA_PATH)
    print("Dataset loaded successfully")

    
    processed_df = process_reviews(df)

    
    processed_df.to_csv(
        OUTPUT_DIR / "processed_reviews.csv",
        index=False
    )
    print("Processed reviews saved")

   
    reviews = processed_df["processed_review"].tolist()

   
    vectorizer = build_tfidf_vectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=1,
        max_df=1.0
    )

    
    tfidf_matrix = vectorizer.fit_transform(reviews)

    
    save_npz(
        OUTPUT_DIR / "tfidf_matrix.npz",
        tfidf_matrix
    )
    print("TF-IDF matrix saved")

    
    vocabulary_data = {
        "vocabulary": vectorizer.vocabulary_,
        "idf": vectorizer.idf_.tolist(),
        "ngram_range": [1, 2],
        "max_features": 5000
    }

    
    with open(
        OUTPUT_DIR / "vocabulary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            vocabulary_data,
            file,
            indent=4
        )

    print("Vocabulary saved")

   
    top_features = get_top_tfidf_features(
        vectorizer,
        tfidf_matrix,
        top_n=20
    )

   
    ngram_results = compare_ngram_features(reviews)

    
    report = []

    report.append("# Day 2 - Feature Report")
    report.append("")

    report.append(f"Total Reviews: {len(reviews)}")
    report.append(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    report.append(f"Vocabulary Size: {len(vectorizer.vocabulary_)}")

    report.append("")
    report.append("## Top TF-IDF Features")
    report.append("")

    report.append("| Feature | Score |")
    report.append("|---|---:|")

    for feature, score in top_features:
        report.append(f"| {feature} | {score:.4f} |")

    report.append("")
    report.append("## N-gram Comparison")
    report.append("")

    report.append("| N-gram | Feature Count |")
    report.append("|---|---:|")

    for name, count in ngram_results.items():
        report.append(f"| {name} | {count} |")

    
    with open(
        OUTPUT_DIR / "feature_report.md",
        "w",
        encoding="utf-8"
    ) as file:
        file.write("\n".join(report))

    
    print("\nDay 2 pipeline completed successfully!")
    print(f"TF-IDF Shape: {tfidf_matrix.shape}")
    print(f"Vocabulary Size: {len(vectorizer.vocabulary_)}")
=======
from pathlib import Path
import pandas as pd
from src.pipeline import preprocess_reviews

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "sample_reviews.csv"

OUTPUT_FILE = BASE_DIR / "data" / "processed_reviews.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    print("Original Reviews:")
    print(df.head())

    processed_df = preprocess_reviews(df)

    processed_df.to_csv(OUTPUT_FILE, index=False)

    print("\nProcessed Reviews:")
    print(
        processed_df[
            ["clean_review", "tokens", "token_count", "filtered_tokens"]
        ].to_string(index=False)
    )

    print("\nTotal Reviews:", len(processed_df))

    print("\nProcessed file saved at:")
    print(OUTPUT_FILE)
>>>>>>> 007e1929c7273184b54fb61fde66be5faa5c048c


if __name__ == "__main__":
    main()
