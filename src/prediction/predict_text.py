import os

import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
vectorizer = joblib.load(os.path.join(MODELS_DIR, "text_vectorizer.pkl"))
text_model = joblib.load(os.path.join(MODELS_DIR, "text_model.pkl"))


def predict_text(text: str) -> tuple[int, float]:
    features = vectorizer.transform([text])
    pred = text_model.predict(features)[0]
    prob = text_model.predict_proba(features)[0][1]
    return int(pred), float(prob)
