
import pandas as pd

from src.preprocess import (
    clean_text,
    tokenize,
    remove_stopwords,
    lemmatize_tokens
)


def process_reviews(df: pd.DataFrame) -> pd.DataFrame:

    if "review" not in df.columns:
        raise ValueError(
            "CSV must contain a 'review' column."
        )

    result = df.copy()

    
    result["clean_review"] = (
        result["review"]
        .fillna("")
        .astype(str)
        .apply(clean_text)
    )

   
    result["tokens"] = (
        result["clean_review"]
        .apply(tokenize)
    )

    
    result["filtered_tokens"] = (
        result["tokens"]
        .apply(remove_stopwords)
    )

    
    result["lemmas"] = (
        result["filtered_tokens"]
        .apply(lemmatize_tokens)
    )

    
    result["processed_review"] = (
        result["lemmas"]
        .apply(lambda words: " ".join(words))
    )

    
    result["token_count"] = (
        result["tokens"]
        .apply(len)
    )

    return result
