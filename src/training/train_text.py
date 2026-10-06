import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SCAM_PATH = os.path.join(BASE_DIR, "data", "scam_calls", "scam.txt")
NOT_SCAM_PATH = os.path.join(BASE_DIR, "data", "scam_calls", "not_scam.txt")
MODELS_PATH = os.path.join(BASE_DIR, "models")

texts = []
labels = []

with open(SCAM_PATH, "r", encoding="utf-8") as file:
    for line in file:
        texts.append(line.strip())
        labels.append(1)

with open(NOT_SCAM_PATH, "r", encoding="utf-8") as file:
    for line in file:
        texts.append(line.strip())
        labels.append(0)

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, labels)

os.makedirs(MODELS_PATH, exist_ok=True)
joblib.dump(vectorizer, os.path.join(MODELS_PATH, "text_vectorizer.pkl"))
joblib.dump(model, os.path.join(MODELS_PATH, "text_model.pkl"))
print("Text model saved!")
