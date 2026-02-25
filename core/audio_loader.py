"""
Audio Loader

Handles loading audio files in WAV, MP3, and M4A/AAC formats.
Normalizes to mono, 16kHz (CREPE-optimal sample rate).

Dependencies: librosa, pydub
"""

import numpy as np
from pathlib import Path


SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".aac", ".flac"}
TARGET_SR = 16000   # Hz — optimal for CREPE pitch detection


def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """
    Load an audio file and return (audio_array, sample_rate).

    - Converts stereo to mono (average channels)
    - Resamples to target_sr
    - Normalizes amplitude to [-1, 1]
    - Supports WAV, MP3, M4A/AAC, FLAC

    Args:
        path: Path to audio file
        target_sr: Target sample rate (default 16000 for CREPE)

    Returns:
        (y, sr) where y is float32 numpy array, sr is sample rate
    """
    import librosa

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {suffix}. "
            f"Supported: {', '.join(SUPPORTED_FORMATS)}"
        )

    # librosa handles WAV, MP3, FLAC natively.
    # For M4A/AAC, it falls back to soundfile/pydub.
    try:
        y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    except Exception as e:
        # Fallback: use pydub to convert to WAV first
        try:
            y, sr = _load_via_pydub(path, target_sr)
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load {path}.\n"
                f"librosa error: {e}\n"
                f"pydub error: {e2}\n"
                f"Try: pip install pydub && brew install ffmpeg"
            )

    # Normalize
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak

    return y.astype(np.float32), sr


def _load_via_pydub(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    """Load audio via pydub (requires ffmpeg for M4A/AAC)."""
    from pydub import AudioSegment
    import io

    audio = AudioSegment.from_file(str(path))
    audio = audio.set_channels(1).set_frame_rate(target_sr)

    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples /= 2 ** (audio.sample_width * 8 - 1)   # normalize to [-1, 1]

    return samples, target_sr


def get_duration(y: np.ndarray, sr: int) -> float:
    """Return audio duration in seconds."""
    return len(y) / sr


def split_into_chunks(
    y: np.ndarray,
    sr: int,
    chunk_sec: float = 30.0,
    overlap_sec: float = 1.0,
) -> list[np.ndarray]:
    """
    Split long audio into overlapping chunks for processing.
    Useful for pieces longer than ~60 seconds.
    """
    chunk_samples = int(chunk_sec * sr)
    overlap_samples = int(overlap_sec * sr)
    step = chunk_samples - overlap_samples

    chunks = []
    start = 0
    while start < len(y):
        end = min(start + chunk_samples, len(y))
        chunks.append(y[start:end])
        if end == len(y):
            break
        start += step

    return chunks
