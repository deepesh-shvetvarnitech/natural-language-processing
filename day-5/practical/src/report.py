from pathlib import Path
from collections import Counter

import pandas as pd
import spacy
import joblib

from src.extraction import (
    load_domain_terms,
    extract_entities,
    extract_custom_entities,
    extract_issues,
    extract_keywords
)
from src.embeddings import load_embedding_model
from src.search import embedding_similar_reviews


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA_DIR / "product_reviews.csv")
    terms = load_domain_terms(str(DATA_DIR / "domain_terms.json"))
    nlp = spacy.load("en_core_web_sm")

    vectorizer = joblib.load(
        ARTIFACT_DIR / "tfidf_vectorizer.joblib"
    )

    embedding_model = load_embedding_model()

    import numpy as np

    embeddings = np.load(
        ARTIFACT_DIR / "review_embeddings.npy"
    )

    reviews = df["review"].astype(str).tolist()

    brand_counts = Counter()
    product_counts = Counter()
    issue_counts = Counter()
    keyword_counts = Counter()

    entity_samples = []
    similar_samples = []

    for review in reviews:
        custom_entities = extract_custom_entities(
            review,
            nlp,
            terms
        )

        issues = extract_issues(
            review,
            nlp,
            terms
        )

        keywords = extract_keywords(
            review,
            vectorizer,
            top_n=10
        )

        for entity in custom_entities:
            if entity["label"] == "BRAND":
                brand_counts[entity["text"]] += 1
            elif entity["label"] == "PRODUCT":
                product_counts[entity["text"]] += 1

        issue_counts.update(issues)
        keyword_counts.update(
            item["keyword"] for item in keywords
        )

        if len(entity_samples) < 5:
            entity_samples.append({
                "review": review,
                "entities": extract_entities(review, nlp)
                    + custom_entities,
                "issues": issues
            })

        if len(similar_samples) < 5:
            similar_samples.append({
                "query": review,
                "results": embedding_similar_reviews(
                    review,
                    reviews,
                    embedding_model,
                    embeddings,
                    top_k=3
                )
            })

    report = [
        "# Day 5 Extraction Quality Report",
        "",
        f"Total reviews processed: {len(reviews)}",
        f"Detected brand mentions: {sum(brand_counts.values())}",
        f"Detected product mentions: {sum(product_counts.values())}",
        f"Detected issue mentions: {sum(issue_counts.values())}",
        "",
        "## Top 10 Keywords",
        ""
    ]

    for keyword, count in keyword_counts.most_common(10):
        report.append(f"- {keyword}: {count}")

    report.extend(["", "## Top 5 Common Issues", ""])

    for issue, count in issue_counts.most_common(5):
        report.append(f"- {issue}: {count}")

    report.extend(["", "## Five Entity Extraction Samples", ""])

    for sample in entity_samples:
        report.append(f"### Review: {sample['review']}")
        report.append(f"Entities: {sample['entities']}")
        report.append(f"Issues: {sample['issues']}")
        report.append("")

    report.extend(["", "## Five Similar Review Samples", ""])

    for sample in similar_samples:
        report.append(f"### Query: {sample['query']}")
        report.append(f"Results: {sample['results']}")
        report.append("")

    report.extend([
        "",
        "## Analysis",
        "",
        "spaCy detects common statistical entities such as organizations, "
        "locations, dates, and money when the model recognizes them.",
        "",
        "Brand and product names often require custom PhraseMatcher or "
        "EntityRuler patterns because the default model may not identify "
        "domain-specific names correctly.",
        "",
        "Extraction errors can occur when a product name is unknown, "
        "an issue is expressed indirectly, or a phrase has different wording.",
        "",
        "Extraction can be improved by adding domain terminology, expanding "
        "rule patterns, collecting labeled examples, and training a custom "
        "NER model.",
        "",
        "A product-quality dashboard can group extracted issues by product "
        "and brand, count negative mentions, and track issue frequency over "
        "time."
    ])

    output_path = REPORT_DIR / "extraction_report.md"
    output_path.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    main()