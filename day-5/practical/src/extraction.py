import json

import spacy
from spacy.matcher import PhraseMatcher

from src.preprocess import preprocess_text
from src.search import embedding_similar_reviews


def load_domain_terms(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_spacy_model():
    return spacy.load("en_core_web_sm")


def extract_entities(text: str, nlp) -> list[dict]:
    doc = nlp(text)

    return [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        }
        for ent in doc.ents
    ]


def build_phrase_matcher(nlp, domain_terms: dict):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    for category in ["brands", "products", "issues"]:
        label = category[:-1].upper()

        patterns = [
            nlp.make_doc(term)
            for term in domain_terms.get(category, [])
        ]

        if patterns:
            matcher.add(label, patterns)

    return matcher


def extract_custom_entities(
    text: str,
    nlp,
    domain_terms: dict
) -> list[dict]:
    doc = nlp(text)
    matcher = build_phrase_matcher(nlp, domain_terms)

    entities = []

    for match_id, start, end in matcher(doc):
        span = doc[start:end]
        label = nlp.vocab.strings[match_id]

        entities.append({
            "text": span.text,
            "label": label,
            "start": span.start_char,
            "end": span.end_char
        })

    unique_entities = {}

    for entity in entities:
        key = (
            entity["start"],
            entity["end"],
            entity["label"]
        )
        unique_entities[key] = entity

    return list(unique_entities.values())


def extract_keywords(
    text: str,
    vectorizer=None,
    top_n: int = 5
) -> list[dict]:
    if not text.strip():
        return []

    if vectorizer is None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000
        )

        matrix = vectorizer.fit_transform(
            [preprocess_text(text)]
        )
    else:
        matrix = vectorizer.transform(
            [preprocess_text(text)]
        )

    feature_names = vectorizer.get_feature_names_out()
    scores = matrix.toarray()[0]

    indices = scores.argsort()[::-1]
    results = []

    for index in indices:
        if scores[index] <= 0:
            continue

        results.append({
            "keyword": str(feature_names[index]),
            "score": round(float(scores[index]), 4)
        })

        if len(results) >= top_n:
            break

    return results


def extract_issues(
    text: str,
    nlp,
    domain_terms: dict
) -> list[str]:

    if not text.strip():
        return []

    doc = nlp(text)
    matcher = build_phrase_matcher(nlp, domain_terms)

    issues = []

    for match_id, start, end in matcher(doc):
        label = nlp.vocab.strings[match_id]

        if label == "ISSUE":
            issues.append(doc[start:end].text.lower())

    lemmas = [
        token.lemma_.lower()
        for token in doc
        if not token.is_punct
    ]

    if "battery" in lemmas and "drain" in lemmas:
        issues.append("battery drain")

    if (
        "camera" in lemmas
        and "quality" in lemmas
        and "poor" in lemmas
    ):
        issues.append("poor camera quality")

    return list(dict.fromkeys(issues))


def analyze_review(
    text: str,
    nlp,
    domain_terms: dict,
    vectorizer=None,
    model=None,
    review_embeddings=None,
    reviews=None,
    top_k: int = 3,
    embedding_model=None
) -> dict:

    from src.sentiment import predict_sentiment

    sentiment_result = None

    if model is not None and vectorizer is not None:
        sentiment_result = predict_sentiment(
            text,
            model,
            vectorizer
        )

    similar_reviews = []

    if (
        embedding_model is not None
        and review_embeddings is not None
        and reviews is not None
    ):
        similar_reviews = embedding_similar_reviews(
            text,
            reviews,
            embedding_model,
            review_embeddings,
            top_k
        )

    return {
        "sentiment": sentiment_result,
        "keywords": extract_keywords(text, vectorizer),
        "entities": (
            extract_entities(text, nlp)
            + extract_custom_entities(text, nlp, domain_terms)
        ),
        "issues": extract_issues(text, nlp, domain_terms),
        "similar_reviews": similar_reviews
    }
