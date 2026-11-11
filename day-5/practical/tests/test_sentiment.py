from src.sentiment import predict_sentiment


class DummyVectorizer:
    def transform(self, texts):
        return [[1, 0]]


class DummyModel:
    def predict(self, features):
        return ["negative"]

    def predict_proba(self, features):
        return [[0.8, 0.2]]


def test_predict_sentiment():
    result = predict_sentiment(
        "The battery drains quickly.",
        DummyModel(),
        DummyVectorizer()
    )

    assert result["sentiment"] == "negative"
    assert result["confidence"] == 0.8