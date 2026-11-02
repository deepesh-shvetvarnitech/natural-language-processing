
from src.cleaning import clean_text
from src.tokenization import (
    tokenize_words,
    tokenize_sentences,
    remove_stopwords
)


def test_clean_text():
    text = "HELLO!!! Visit https://example.com"
    result = clean_text(text)

    assert result == "hello visit"


def test_clean_text_removes_html():
    text = "<p>Hello World</p>"
    result = clean_text(text)

    assert result == "hello world"


def test_tokenize_words():
    text = "Hello world"
    result = tokenize_words(text)

    assert result == ["Hello", "world"]


def test_tokenize_sentences():
    text = "I love NLP. Python is amazing."
    result = tokenize_sentences(text)

    assert len(result) == 2


def test_remove_stopwords():
    tokens = ["this", "product", "is", "not", "good"]

    result = remove_stopwords(tokens)

    assert "is" not in result
    assert "not" in result
    assert "product" in result
    assert "good" in result


def test_empty_text():
    assert tokenize_words("") == []
    assert tokenize_sentences("") == []
    assert clean_text("") == ""
