"""
Pitch Extractor — CQT + PESTO backends
"""

import numpy as np
import torch
import pesto
import librosa
from dataclasses import dataclass
from scipy.signal import medfilt
from .config import PITCH_EXTRACTION


@dataclass
class PitchTrack:
    """Result of pitch extraction."""
    times: np.ndarray
    frequencies: np.ndarray
    confidences: np.ndarray
    sample_rate: int
    hop_size: int

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_size

    @property
    def duration_sec(self) -> float:
        return self.times[-1] if len(self.times) > 0 else 0.0

    @property
    def voiced_mask(self) -> np.ndarray:
        return self.frequencies > 0

    def voiced_frequencies(self) -> np.ndarray:
        return self.frequencies[self.voiced_mask]

    def voiced_times(self) -> np.ndarray:
        return self.times[self.voiced_mask]

    def voiced_confidences(self) -> np.ndarray:
        return self.confidences[self.voiced_mask]


PESTO_SR = 22050
HOP_SIZE_MS = 10.0
HOP_SIZE_SEC = HOP_SIZE_MS / 1000.0
CONFIDENCE_THRESHOLD = 0.4


def extract_pitch_pesto(
    y: np.ndarray,
    sr: int,
    fmin: float = 80.0,
    fmax: float = 1200.0,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    verbose: bool = True,
) -> PitchTrack:
    """Extract pitch using PESTO."""
    if verbose:
        print("      Running PESTO pitch detection...")

    if sr != PESTO_SR:
        if verbose:
            print(f"      Resampling from {sr}Hz to {PESTO_SR}Hz for PESTO...")
        y = librosa.resample(y, orig_sr=sr, target_sr=PESTO_SR)
        sr = PESTO_SR

    audio_tensor = torch.from_numpy(y).float()
    hop_length = int(HOP_SIZE_SEC * sr)

    timesteps, frequencies, confidence, _ = pesto.predict(
        audio_tensor,
        sr=sr,
        step_size=10.0,
        convert_to_freq=True,
        num_chunks=4,
        inference_mode=True,
    )

    times = timesteps.numpy() if hasattr(timesteps, 'numpy') else np.array(timesteps)
    if len(times) > 1 and times[-1] > 10000:
        times = times / 1000.0
    freqs = frequencies.numpy() if hasattr(frequencies, 'numpy') else np.array(frequencies)
    confs = confidence.numpy() if hasattr(confidence, 'numpy') else np.array(confidence)

    freqs = freqs.copy()
    freqs[confs < confidence_threshold] = 0.0

    out_of_range = (freqs > 0) & ((freqs < fmin) | (freqs > fmax))
    freqs[out_of_range] = 0.0

    if verbose:
        voiced = freqs > 0
        voiced_ratio = voiced.sum() / len(voiced)
        voiced_freqs = freqs[voiced]
        print(f"      PESTO complete. Voiced frames: {voiced_ratio:.1%}")
        if len(voiced_freqs) > 0:
            print(f"      Frequency range: {voiced_freqs.min():.1f} - {voiced_freqs.max():.1f} Hz")
            print(f"      Median pitch: {np.median(voiced_freqs):.1f} Hz")

    return PitchTrack(
        times=times,
        frequencies=freqs,
        confidences=confs,
        sample_rate=sr,
        hop_size=hop_length,
    )


def extract_pitch_cqt(
        y: np.ndarray,
        sr: int,
        fmin: float = 80.0,
        fmax: float = 1200.0,
        confidence_threshold: float = PITCH_EXTRACTION['cqt_confidence_threshold'],
        energy_threshold_db: float = PITCH_EXTRACTION['cqt_energy_threshold_db'],
        verbose: bool = True,
) -> PitchTrack:
    """
    Extract pitch using CQT with 53-EDO resolution for microtones.
    """
    if verbose:
        print("      Running CQT pitch detection (53-EDO)...")

    hop_length = PITCH_EXTRACTION['hop_length']
    bins_per_octave = PITCH_EXTRACTION['bins_per_octave']

    # Calculate maximum frequency we can analyze (Nyquist limit)
    nyquist = sr / 2
    # Set fmax to be safe (a bit below Nyquist)
    fmax_cqt = min(fmax, nyquist * 0.95)

    # Calculate number of octaves needed
    n_octaves = int(np.ceil(np.log2(fmax_cqt / fmin))) + 1

    if verbose:
        print(f"      Sample rate: {sr}Hz, Nyquist: {nyquist}Hz")
        print(f"      CQT range: {fmin:.0f} - {fmax_cqt:.0f} Hz, Octaves: {n_octaves}")

    C = np.abs(librosa.cqt(
        y,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave,
        n_bins=n_octaves * bins_per_octave,
        fmin=fmin,
    ))

    freqs = librosa.cqt_frequencies(
        n_bins=C.shape[0],
        fmin=fmin,
        bins_per_octave=bins_per_octave
    )

    max_bins = np.argmax(C, axis=0)
    pitch_freqs = freqs[max_bins]

    max_vals = np.max(C, axis=0)
    mean_vals = np.mean(C, axis=0)
    confidence = max_vals / (mean_vals + 1e-6)
    confidence = np.clip(confidence, 0, 1)

    # Energy-based silence detection
    frame_energy = np.sum(C, axis=0)
    frame_energy_db = 20 * np.log10(frame_energy + 1e-10)
    silence_frames = frame_energy_db < energy_threshold_db

    # Filter low confidence frames
    low_confidence = confidence < confidence_threshold
    pitch_freqs[low_confidence] = 0.0
    confidence[low_confidence] = 0.0

    # Filter silence frames
    pitch_freqs[silence_frames] = 0.0
    confidence[silence_frames] = 0.0

    # Filter out-of-range frequencies
    out_of_range = (pitch_freqs > 0) & ((pitch_freqs < fmin) | (pitch_freqs > fmax))
    pitch_freqs[out_of_range] = 0.0
    confidence[out_of_range] = 0.0

    times = np.arange(len(pitch_freqs)) * hop_length / sr

    if verbose:
        voiced = pitch_freqs > 0
        voiced_ratio = voiced.sum() / len(voiced)
        silence_ratio = silence_frames.sum() / len(silence_frames)
        print(f"      CQT complete. Voiced frames: {voiced_ratio:.1%}")
        print(f"      Silence frames: {silence_ratio:.1%}")
        print(f"      Using {bins_per_octave} bins/octave for microtonal resolution")

    return PitchTrack(
        times=times,
        frequencies=pitch_freqs,
        confidences=confidence,
        sample_rate=sr,
        hop_size=hop_length,
    )

def preserve_ornament_contours(track: PitchTrack) -> PitchTrack:
    """Detect and preserve rapid pitch changes that might be ornaments."""
    freqs = track.frequencies.copy()
    confs = track.confidences

    diff = np.diff(np.concatenate(([freqs[0]], freqs)))
    change_rate = np.abs(diff)

    ornament_mask = change_rate > 5
    ornament_mask = ornament_mask & (confs > 0.3)

    smoothed = freqs.copy()
    i = 0
    while i < len(freqs):
        if ornament_mask[i]:
            start = i
            while i < len(freqs) and ornament_mask[i]:
                i += 1
            end = i
            smoothed[start:end] = freqs[start:end]
        else:
            i += 1

    return PitchTrack(
        times=track.times,
        frequencies=smoothed,
        confidences=track.confidences,
        sample_rate=track.sample_rate,
        hop_size=track.hop_size,
    )


def extract_pitch_pesto_enhanced(
    y: np.ndarray,
    sr: int,
    fmin: float = 80.0,
    fmax: float = 1200.0,
    preserve_ornaments: bool = True,
    verbose: bool = True,
) -> PitchTrack:
    """Enhanced pitch extraction that preserves ornament details."""
    track = extract_pitch_pesto(y, sr, fmin, fmax, verbose)

    if preserve_ornaments:
        track = preserve_ornament_contours(track)

    return track


def extract_pitch_hybrid(
        y: np.ndarray,
        sr: int,
        fmin: float = 80.0,
        fmax: float = 1200.0,
        verbose: bool = True,
) -> PitchTrack:
    """Try CQT first, fall back to PESTO if confidence is low"""
    cqt_track = extract_pitch_cqt(y, sr, fmin, fmax, verbose=verbose)

    voiced_ratio = cqt_track.voiced_mask.sum() / len(cqt_track.voiced_mask)

    if voiced_ratio > 0.4:
        if verbose:
            print(f"      Using CQT results (voiced: {voiced_ratio:.1%})")
        return cqt_track
    else:
        if verbose:
            print(f"      CQT voiced ratio low ({voiced_ratio:.1%}), falling back to PESTO...")
        return extract_pitch_pesto_enhanced(y, sr, fmin, fmax, True, verbose)


def simple_segment_notes(
    frequencies,
    times=None,
    time_step=0.032,
    min_duration_sec=0.12,
    tolerance_hz=6.0,
    **kwargs
):
    """
    Segments raw pitch tracking into discrete notes,
    filtering out vibrato and CQT noise.
    """
    # --- NEW FIX: Unpack the PitchTrack object if passed directly ---
    if hasattr(frequencies, 'frequencies'):
        # If the object has times attached to it, grab them too
        if times is None and hasattr(frequencies, 'times'):
            times = frequencies.times
        # Extract the actual list of numbers from the object
        frequencies = frequencies.frequencies
    # ----------------------------------------------------------------

    if len(frequencies) == 0:
        return []

    # If we STILL don't have times, build them mathematically
    if times is None:
        times = np.arange(len(frequencies)) * time_step

    # 1. VIBRATO SMOOTHING
    freqs_array = np.array(frequencies)
    smoothed_freqs = medfilt(freqs_array, kernel_size=9)

    segments = []
    current_note_start = times[0]
    current_freqs = [smoothed_freqs[0]]

    # 2. SEGMENTATION WITH TOLERANCE
    for i in range(1, len(smoothed_freqs)):
        freq = smoothed_freqs[i]

        prev_freq = np.mean(current_freqs) if len(current_freqs) > 0 else 0

        # Break the note IF it jumps outside our Hz tolerance, OR hits silence
        is_pitch_jump = abs(freq - prev_freq) > tolerance_hz
        is_silence = freq <= 0

        if is_pitch_jump or is_silence:
            duration = times[i] - current_note_start

            # 3. DURATION GATE
            if duration >= min_duration_sec and prev_freq > 0:
                segments.append({
                    "start_time": current_note_start,
                    "duration_sec": duration,
                    "median_freq": np.median(current_freqs),
                    "mean_confidence": 1.0
                })

            current_note_start = times[i]
            current_freqs = []

        if freq > 0:
            current_freqs.append(freq)

    # Catch the final trailing note
    if len(current_freqs) > 0:
        duration = times[-1] - current_note_start
        if duration >= min_duration_sec and np.mean(current_freqs) > 0:
            segments.append({
                "start_time": current_note_start,
                "duration_sec": duration,
                "median_freq": np.median(current_freqs),
                "mean_confidence": 1.0
            })

    return segments

# Aliases
extract_pitch_crepe = extract_pitch_pesto
extract_pitch_pyin = extract_pitch_pesto
extract_pitch_default = extract_pitch_cqt  # CQT is default!
