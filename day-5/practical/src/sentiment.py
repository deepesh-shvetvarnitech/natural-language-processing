import joblib
from pathlib import Path


ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def load_sentiment_model():
    model_path = ARTIFACT_DIR / "sentiment_model.joblib"
    vectorizer_path = ARTIFACT_DIR / "tfidf_vectorizer.joblib"

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer


def predict_sentiment(text: str, model, vectorizer) -> dict:
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities))
    else:
        confidence = 1.0

    return {
        "sentiment": str(prediction),
        "confidence": round(confidence, 4)
    }