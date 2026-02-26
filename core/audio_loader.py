"""
Audio Loader — SOTA Pipeline

Handles loading audio files, applying spectral denoising, and optionally
isolating the melody with HPSS to provide the cleanest possible signal
for pitch tracking.
"""

import numpy as np
from pathlib import Path
import librosa
import noisereduce as nr

SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".aac", ".flac"}
TARGET_SR = 16000   # Hz — Native rate for PENN pitch extraction


def load_audio(path: str | Path, target_sr: int = TARGET_SR, isolate_melody: bool = False) -> tuple[np.ndarray, int]:
    """
    Load an audio file and return a pre-processed (audio_array, sample_rate).

    - Converts stereo to mono and resamples to target_sr.
    - Applies spectral denoising to remove background hiss/hum.
    - (Optional) Applies HPSS to remove percussive elements.
    - Normalizes amplitude to [-1, 1].

    Args:
        path: Path to audio file.
        target_sr: Target sample rate.
        isolate_melody: If True, uses HPSS to remove percussion/noise.

    Returns:
        (y, sr) where y is a float32 numpy array and sr is the sample rate.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {path.suffix}. Supported: {SUPPORTED_FORMATS}")

    try:
        y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    except Exception as e:
        # Fallback for formats librosa doesn't handle natively
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(str(path)).set_channels(1).set_frame_rate(target_sr)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            y = samples / (2**(audio.sample_width * 8 - 1))
            sr = target_sr
        except Exception as e2:
            raise RuntimeError(f"Failed to load {path} with both librosa and pydub. Error: {e2}")

    # 1. Apply SOTA Spectral Denoising
    y = nr.reduce_noise(y=y, sr=sr, stationary=True)

    # 2. Apply SOTA Source Separation if requested
    if isolate_melody:
        y, _ = librosa.effects.hpss(y)

    # 3. Normalize amplitude
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak

    return y.astype(np.float32), sr


def get_duration(y: np.ndarray, sr: int) -> float:
    """Return audio duration in seconds."""
    return len(y) / sr
