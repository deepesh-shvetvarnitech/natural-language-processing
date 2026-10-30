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


if __name__ == "__main__":
    main()
