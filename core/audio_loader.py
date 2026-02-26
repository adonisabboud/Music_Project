"""
Audio Loader — SOTA Pipeline

Handles loading audio files in WAV, MP3, and M4A/AAC formats.
Normalizes to mono, 16kHz (Native sample rate for PENN).

Includes optional Harmonic-Percussive Source Separation (HPSS)
to strip away drums, plectrum clicks, and room noise.
"""

import numpy as np
from pathlib import Path


SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".aac", ".flac"}
TARGET_SR = 16000   # Hz — Native rate for PENN pitch extraction


def load_audio(path: str | Path, target_sr: int = TARGET_SR, isolate_melody: bool = False) -> tuple[np.ndarray, int]:
    """
    Load an audio file and return (audio_array, sample_rate).

    - Converts stereo to mono
    - Resamples to target_sr
    - Normalizes amplitude to [-1, 1]
    - (Optional) Applies HPSS to isolate the harmonic melody

    Args:
        path: Path to audio file
        target_sr: Target sample rate (default 16000 for PENN)
        isolate_melody: If True, uses HPSS to remove percussion/noise

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

    try:
        y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    except Exception as e:
        try:
            y, sr = _load_via_pydub(path, target_sr)
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load {path}.\n"
                f"librosa error: {e}\n"
                f"pydub error: {e2}\n"
                f"Try: pip install pydub && brew install ffmpeg"
            )

    # Apply SOTA Source Separation if requested
    if isolate_melody:
        # HPSS splits the audio into Harmonic (Melody) and Percussive (Noise/Drums)
        # We only keep the harmonic component for cleaner pitch tracking
        y, _ = librosa.effects.hpss(y)

    # Normalize
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak

    return y.astype(np.float32), sr


def _load_via_pydub(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    """Load audio via pydub (requires ffmpeg for M4A/AAC)."""
    from pydub import AudioSegment

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
    Useful for pieces longer than ~60 seconds to prevent RAM overload.
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