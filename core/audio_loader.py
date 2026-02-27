"""
Audio Loader — SOTA Pipeline

Handles loading audio files and applying spectral denoising to provide
a clean signal for pitch tracking, while preserving the natural attacks
needed for onset detection.
"""

import numpy as np
from pathlib import Path
import librosa
import noisereduce as nr
import soundfile as sf


SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".aac", ".flac"}
TARGET_SR = 16000   # Hz — Native rate for PENN pitch extraction


def load_audio(
    path: str | Path, 
    target_sr: int = TARGET_SR, 
) -> tuple[np.ndarray, int]:
    """
    Load an audio file and return a pre-processed (audio_array, sample_rate).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {path.suffix}. Supported: {SUPPORTED_FORMATS}")

    try:
        y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    except Exception as e:
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(str(path)).set_channels(1).set_frame_rate(target_sr)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            y = samples / (2**(audio.sample_width * 8 - 1))
            sr = target_sr
        except Exception as e2:
            raise RuntimeError(f"Failed to load {path} with both librosa and pydub. Error: {e2}")


    # 2. Normalize amplitude
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak

    return y.astype(np.float32), sr


def get_duration(y: np.ndarray, sr: int) -> float:
    """Return audio duration in seconds."""
    return len(y) / sr


def save_debug_audio(y: np.ndarray, sr: int, path: str | Path):
    """Saves a NumPy audio array to a WAV file for debugging."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr)
