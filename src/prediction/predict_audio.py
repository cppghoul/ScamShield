import os
import sys

import joblib

from src.common.audio_features import extract_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "audio_model.pkl")

if len(sys.argv) < 2:
    print("Usage: python -m src.prediction.predict_audio path_to_file.wav")
    sys.exit(1)

file_path = sys.argv[1]
model = joblib.load(MODEL_PATH)
features = extract_features(file_path)
prediction = model.predict([features])[0]
prob = model.predict_proba([features])[0][1]

if prediction == 1:
    print(f"FAKE voice detected ({prob * 100:.2f}%)")
else:
    print(f"REAL voice ({(1 - prob) * 100:.2f}%)")
