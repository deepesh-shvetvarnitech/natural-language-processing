from src.search import search_tfidf

def test_tfidf_returns_results(sample_df, sample_tfidf):
    vectorizer, matrix = sample_tfidf
    results = search_tfidf("battery", sample_df, vectorizer, matrix, top_k=2)
    assert len(results) == 2
    assert results[0]["review_id"] in {"A", "C"}
    assert results[0]["search_type"] == "tfidf"

def test_exclude_id(sample_df, sample_tfidf):
    vectorizer, matrix = sample_tfidf
    results = search_tfidf("battery", sample_df, vectorizer, matrix, top_k=3, exclude_review_id="A")
    assert all(item["review_id"] != "A" for item in results)
