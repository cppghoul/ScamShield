import warnings

import librosa
import numpy as np
import whisper

print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper loaded.")


def _transcribe_with_array(file_path: str) -> str:
    audio, _ = librosa.load(file_path, sr=16000, mono=True)
    if audio.size == 0:
        return ""
    audio = np.asarray(audio, dtype=np.float32)
    result = whisper_model.transcribe(audio)
    return result.get("text", "")


def transcribe_audio(file_path: str) -> str:
    try:
        return _transcribe_with_array(file_path)
    except Exception as array_exc:
        warnings.warn(
            f"Librosa decode failed ({array_exc}); fallback to whisper file decoding.",
            RuntimeWarning,
        )
        result = whisper_model.transcribe(file_path)
        return result["text"]
