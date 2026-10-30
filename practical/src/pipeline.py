
import pandas as pd

from src.cleaning import clean_text
from src.tokenization import (
    tokenize_words,
    tokenize_sentences,
    remove_stopwords
)


def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    

    
    if "review" not in df.columns:
        raise ValueError("CSV must contain a 'review' column")

    
    df = df.copy()

    
    df["review"] = df["review"].fillna("")

   
    df["clean_review"] = df["review"].apply(clean_text)

    
    df["tokens"] = df["clean_review"].apply(tokenize_words)

    
    df["token_count"] = df["tokens"].apply(len)

    
    df["sentences"] = df["review"].apply(
    tokenize_sentences
)

    
    df["filtered_tokens"] = df["tokens"].apply(
        remove_stopwords
    )

    return df
