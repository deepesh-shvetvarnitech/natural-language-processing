
import pytest

from src.preprocess import (
    clean_text,
    tokenize,
    remove_stopwords,
    stem_tokens,
    lemmatize_tokens,
    get_pos_tags,
    preprocess
)


def test_clean_text():

    result = clean_text(
        "<b>Hello</b> WORLD!"
    )

    assert result == "hello world"


def test_tokenize():

    result = tokenize("I love NLP")

    assert result == ["I", "love", "NLP"]


def test_remove_stopwords():

    result = remove_stopwords(
        ["this", "is", "not", "good"]
    )

    assert "this" not in result
    assert "not" in result


def test_stemming():

    result = stem_tokens(
        ["playing", "played"]
    )

    assert result == ["play", "play"]


def test_lemmatization():

    result = lemmatize_tokens(
        ["cars", "running"]
    )

    assert len(result) == 2


def test_pos_tagging():

    result = get_pos_tags(
        "The movie is excellent"
    )

    assert len(result) > 0

    assert all(
        isinstance(item, tuple)
        for item in result
    )


def test_stemming_and_lemmatization_error():

    with pytest.raises(ValueError):

        preprocess(
            "This is a good movie",
            use_stemming=True,
            use_lemmatization=True
        )
