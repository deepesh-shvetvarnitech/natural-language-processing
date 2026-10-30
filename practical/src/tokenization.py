import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


def tokenize_words(text: str) -> list:
    if not isinstance(text, str) or not text.strip():
        return []

    return word_tokenize(text)


def tokenize_sentences(text: str) -> list:
    if not isinstance(text, str) or not text.strip():
        return []

    return sent_tokenize(text)


def remove_stopwords(tokens: list, preserve_negation: bool = True) -> list:
    stop_words = set(stopwords.words("english"))
    negation_words = {"no", "not", "nor"}

    if preserve_negation:
        stop_words = stop_words - negation_words

    filtered_tokens = [
        token for token in tokens if token.lower() not in stop_words
    ]

    return filtered_tokens
