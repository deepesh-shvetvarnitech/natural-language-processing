
import pandas as pd

<<<<<<< HEAD
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
=======
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
>>>>>>> 007e1929c7273184b54fb61fde66be5faa5c048c
