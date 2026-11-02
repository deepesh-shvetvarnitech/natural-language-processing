
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(
    ngram_range=(1, 1),
    max_features=5000,
    min_df=2,
    max_df=0.95
):
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        lowercase=False
    )

    return vectorizer


def convert_reviews_to_tfidf(reviews):
    vectorizer = build_tfidf_vectorizer()
    matrix = vectorizer.fit_transform(reviews)

    return vectorizer, matrix


def get_top_tfidf_features(vectorizer, matrix, top_n=20):
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    if matrix.shape[1] == 0:
        return []

    average_scores = matrix.mean(axis=0).A1
    feature_names = vectorizer.get_feature_names_out()

    top_indices = average_scores.argsort()[::-1][:top_n]

    return [
        (feature_names[index], float(average_scores[index]))
        for index in top_indices
    ]


def compare_ngram_features(reviews):
    results = {}

    ngram_options = {
        "Unigram": (1, 1),
        "Bigram": (2, 2),
        "Trigram": (3, 3),
        "Unigram + Bigram": (1, 2),
        "Unigram + Bigram + Trigram": (1, 3)
    }

    for name, ngram_range in ngram_options.items():

        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=1,
            max_df=1.0
        )

        matrix = vectorizer.fit_transform(reviews)

        results[name] = {
            "feature_count": matrix.shape[1],
            "features": vectorizer.get_feature_names_out().tolist()
        }

    return results
