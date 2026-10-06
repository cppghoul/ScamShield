import os
import sys

import joblib

from src.common.audio_features import extract_features
from src.prediction.predict_text import predict_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
audio_model = joblib.load(os.path.join(MODELS_DIR, "audio_model.pkl"))


def speech_to_text(file_path: str) -> str:
    try:
        import whisper
    except ImportError as error:
        raise ImportError("Установите whisper: pip install openai-whisper") from error

    model = whisper.load_model("small")
    result = model.transcribe(file_path)
    return result["text"]


def predict_audio(file_path: str) -> tuple[int, float]:
    features = extract_features(file_path)
    pred = audio_model.predict([features])[0]
    prob = audio_model.predict_proba([features])[0][1]
    return int(pred), float(prob)


def main(file_path: str) -> None:
    print(f"\nAnalyzing file: {file_path}\n")
    audio_pred, audio_prob = predict_audio(file_path)
    audio_label = "FAKE" if audio_pred == 1 else "REAL"
    print(f"Audio analysis: {audio_label} (prob: {audio_prob * 100:.1f}%)")
    text = speech_to_text(file_path)
    print(f"\nTranscribed text:\n{text}")
    text_pred, text_prob = predict_text(text)
    text_label = "SCAM" if text_pred == 1 else "NOT SCAM"
    print(f"Text analysis: {text_label} (prob: {text_prob * 100:.1f}%)")
    final = "⚠️ ALERT: POSSIBLE FRAUD" if audio_pred == 1 or text_pred == 1 else "✅ Likely Safe"
    print(f"\nFINAL VERDICT: {final}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.prediction.predict_full <audio_file.wav>")
        sys.exit(1)
    audio_file = sys.argv[1]
    main(audio_file)
