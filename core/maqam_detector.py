"""
Maqam Auto-Detector

This module uses a State-of-the-Art "tuning-invariant" method to identify
the maqam from a pitch track. It works by:
1. Creating a "fingerprint" of the audio's pitch distribution using KDE.
2. Creating a "fingerprint" for each theoretical maqam scale.
3. Using circular cross-correlation (via FFT) to find the optimal tuning
   offset that best aligns the audio to each maqam.
4. Scoring each maqam at its optimal tuning, making the detection robust
   to recordings that are not tuned to a perfect A4=440Hz.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.fft import rfft, irfft
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

from .tuning import (
    MAQAM,
    OCT,
    build_scale,
    ScaleNote,
)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MICRO_WEIGHT = 3.0
NORMAL_WEIGHT = 1.0
TONIC_BONUS = 1.2
PEAK_SIGMA_CENTS = 5.0

@dataclass
class MaqamCandidate:
    name: str
    score: float
    tuning_offset_cents: float
    scale: list[ScaleNote]

    def __repr__(self):
        return (f"MaqamCandidate({self.name!r}, score={self.score:.3f}, "
                f"tuning={self.tuning_offset_cents:+.1f}c)")

# ──────────────────────────────────────────────
# SOTA Fingerprinting and Cross-Correlation
# ──────────────────────────────────────────────

def _create_maqam_fingerprint(scale: list[ScaleNote], length: int = 1200) -> np.ndarray:
    """
    Generates a 1200-point ideal "fingerprint" for a maqam scale.
    """
    fingerprint = np.zeros(length)
    
    for note in scale:
        expected_cents = (note.abs_commas % OCT) * (1200.0 / OCT)
        center_bin = int(round(expected_cents)) % length
        
        weight = MICRO_WEIGHT if note.is_micro else NORMAL_WEIGHT
        if note.is_tonic:
            weight *= TONIC_BONUS

        x = np.arange(length)
        dist = np.min([np.abs(x - center_bin), 
                       np.abs(x - center_bin - length), 
                       np.abs(x - center_bin + length)], axis=0)
        
        peak = weight * np.exp(-0.5 * (dist / PEAK_SIGMA_CENTS)**2)
        fingerprint += peak
        
    return fingerprint / np.sum(fingerprint)


def detect_maqam_sota(
        frequencies: list[float],
        confidences: Optional[list[float]] = None,
        top_n: int = 3
) -> list[MaqamCandidate]:
    """
    Detects Maqam using tuning-invariant cross-correlation.
    """
    if confidences is None:
        confidences = [1.0] * len(frequencies)

    _, _, audio_fp = extract_tuning_peaks_kde(frequencies, confidences)
    if np.sum(audio_fp) == 0:
        return []
    
    audio_fp /= np.sum(audio_fp)
    audio_fft = rfft(audio_fp)
    
    candidates = []

    for maqam_name, m in MAQAM.items():
        for upper_j in m.upper_options:
            scale = build_scale(maqam_name, upper_j)
            maqam_fp = _create_maqam_fingerprint(scale)
            maqam_fft = rfft(maqam_fp)
            cross_corr = irfft(audio_fft * np.conj(maqam_fft))

            score = np.max(cross_corr)
            offset_cents = np.argmax(cross_corr)
            
            if offset_cents > 600:
                offset_cents -= 1200

            name = f"{maqam_name} (upper {upper_j})" if upper_j != m.upper_default else maqam_name

            candidates.append(MaqamCandidate(
                name=name, 
                score=score,
                tuning_offset_cents=float(offset_cents),
                scale=scale
            ))

    max_score = max(c.score for c in candidates) if candidates else 1.0
    for c in candidates:
        c.score /= max_score

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_n]


def extract_tuning_peaks_kde(
        frequencies: list[float],
        confidences: list[float],
        min_conf: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SOTA Dynamic Tuning using Kernel Density Estimation (KDE).
    """
    valid_idx = [i for i, (f, c) in enumerate(zip(frequencies, confidences)) if f > 0 and c > min_conf]
    if not valid_idx:
        return np.array([]), np.array([]), np.zeros(1200)

    valid_freqs = np.array([frequencies[i] for i in valid_idx])
    valid_confs = np.array([confidences[i] for i in valid_idx])

    cents = 1200 * np.log2(valid_freqs / 261.63)
    cents_folded = np.mod(cents, 1200)

    cents_padded = np.concatenate([cents_folded - 1200, cents_folded, cents_folded + 1200])
    weights_padded = np.concatenate([valid_confs, valid_confs, valid_confs])

    kde = gaussian_kde(cents_padded, weights=weights_padded, bw_method=0.02)

    x_grid = np.arange(1200)
    kde_curve = kde(x_grid)
    
    peaks, _ = find_peaks(kde_curve, distance=35, prominence=0.05)

    return peaks, kde_curve[peaks], kde_curve


def detect_maqam_with_consistency(
        frequencies: list[float],
        confidences: Optional[list[float]] = None,
) -> list[MaqamCandidate]:
    """
    Main entry point for maqam detection.
    """
    return detect_maqam_sota(frequencies, confidences, top_n=5)
