import spacy

from src.extraction import (
    load_domain_terms,
    extract_custom_entities,
    extract_issues,
    extract_entities
)


nlp = spacy.load("en_core_web_sm")
terms = load_domain_terms("data/domain_terms.json")


def test_custom_brand_extraction():
    text = "The OnePlus battery drains quickly after the latest update."
    entities = extract_custom_entities(text, nlp, terms)

    assert any(
        entity["text"].lower() == "oneplus"
        and entity["label"] == "BRAND"
        for entity in entities
    )


def test_issue_extraction():
    text = "Battery drains very quickly."
    issues = extract_issues(text, nlp, terms)

    assert "battery drain" in issues


def test_camera_issue_extraction():
    text = "Camera quality is poor."
    issues = extract_issues(text, nlp, terms)

    assert "poor camera quality" in issues


def test_standard_entities():
    text = "I bought a Samsung Galaxy S24 in Mumbai for ₹74,999."
    entities = extract_entities(text, nlp)

    assert isinstance(entities, list)