import os

import joblib
import numpy as np

from src.common.audio_features import extract_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TEXT_MODEL_PATH = os.path.join(BASE_DIR, "models", "text_model.pkl")
TEXT_VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "text_vectorizer.pkl")
AUDIO_MODEL_PATH = os.path.join(BASE_DIR, "models", "audio_model.pkl")

print("Loading models...")
text_model = joblib.load(TEXT_MODEL_PATH)
text_vectorizer = joblib.load(TEXT_VECTORIZER_PATH)
audio_model = joblib.load(AUDIO_MODEL_PATH)
print("Models loaded successfully.")


def analyze_text(text: str) -> float:
    vector = text_vectorizer.transform([text])
    prob = text_model.predict_proba(vector)[0][1]
    return float(prob)


def extract_audio_features(file_path: str) -> np.ndarray:
    return extract_features(file_path)


def analyze_audio(file_path: str) -> float:
    features = extract_audio_features(file_path)
    prob = audio_model.predict_proba([features])[0][1]
    return float(prob)


def analyze_full(text: str, audio_path: str | None = None) -> tuple[float, float, float | None]:
    text_prob = analyze_text(text)
    if audio_path:
        audio_prob = analyze_audio(audio_path)
        final_prob = (text_prob + audio_prob) / 2
        return final_prob, text_prob, audio_prob
    return text_prob, text_prob, None
