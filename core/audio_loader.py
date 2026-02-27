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
from scipy.signal import lfilter


SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".aac", ".flac"}
TARGET_SR = 16000   # Hz — Native rate for PENN pitch extraction


def apply_lpc_inverse_filter(y: np.ndarray, sr: int, order: int = 12) -> np.ndarray:
    """
    Applies an LPC inverse filter to flatten the spectral envelope of the audio.
    This removes the resonant characteristics of an instrument's body,
    preventing loud harmonics from being mistaken for the fundamental pitch.
    """
    # Estimate the LPC filter coefficients from the audio signal
    # These coefficients model the resonant spectral envelope.
    a = librosa.lpc(y, order=order)
    
    # Apply the inverse of this filter to the signal.
    # This "flattens" the spectrum, removing the resonant peaks.
    y_flat = lfilter(a, [1], y)
    
    return y_flat


def load_audio(path: str | Path, target_sr: int = TARGET_SR, isolate_melody: bool = False) -> tuple[np.ndarray, int]:
    """
    Load an audio file and return a pre-processed (audio_array, sample_rate).

    - Converts stereo to mono and resamples to target_sr.
    - Applies LPC inverse filtering to remove instrument resonance.
    - Applies spectral denoising to remove background hiss/hum.
    - (Optional) Applies HPSS to remove percussive elements.
    - Normalizes amplitude to [-1, 1].
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

    # 1. Apply LPC Inverse Filtering to flatten spectral envelope
    y = apply_lpc_inverse_filter(y, sr)

    # 2. Apply SOTA Spectral Denoising
    y = nr.reduce_noise(y=y, sr=sr, stationary=True)

    # 3. Apply SOTA Source Separation if requested
    if isolate_melody:
        y, _ = librosa.effects.hpss(y)

    # 4. Normalize amplitude
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak

    return y.astype(np.float32), sr


def get_duration(y: np.ndarray, sr: int) -> float:
    """Return audio duration in seconds."""
    return len(y) / sr
