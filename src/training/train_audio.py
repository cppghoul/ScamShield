import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.audio_features import extract_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
REAL_PATH = os.path.join(BASE_DIR, "data", "real")
FAKE_PATH = os.path.join(BASE_DIR, "data", "fake")
MODELS_PATH = os.path.join(BASE_DIR, "models")

X = []
y = []

print("Loading real files...")
for file_name in os.listdir(REAL_PATH):
    if file_name.endswith(".wav"):
        features = extract_features(os.path.join(REAL_PATH, file_name))
        X.append(features)
        y.append(0)

print("Loading fake files...")
for file_name in os.listdir(FAKE_PATH):
    if file_name.endswith(".wav"):
        features = extract_features(os.path.join(FAKE_PATH, file_name))
        X.append(features)
        y.append(1)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

os.makedirs(MODELS_PATH, exist_ok=True)
joblib.dump(pipeline, os.path.join(MODELS_PATH, "audio_model.pkl"))
print("Model saved!")
