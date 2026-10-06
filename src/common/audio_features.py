import librosa
import numpy as np


def extract_features(file_path: str) -> np.ndarray:
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))
    features = np.hstack([mfcc_mean, spectral_centroid, zero_crossing])
    return features
