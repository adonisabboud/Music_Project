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

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.fft import rfft, irfft
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from .config import PITCH_EXTRACTION, MAQAM_DETECTION
from .tuning import (
    MAQAM,
    OCT,
    build_scale,
    ScaleNote,
    comma_to_freq,
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
        top_n: int = 3,
        hop_size_sec: float = 0.01,
) -> list[MaqamCandidate]:
    """
    Detects Maqam using tuning-invariant cross-correlation.
    Clamped to +/- 100 cents to prevent extreme transposition hallucinations.
    """
    if confidences is None:
        confidences = [1.0] * len(frequencies)

    tonic_hz, tonic_confidence = detect_tonic(frequencies, confidences, hop_size_sec)

    _, _, audio_fp = extract_tuning_peaks_kde(frequencies, confidences)
    if np.sum(audio_fp) == 0:
        return []

    audio_fp /= np.sum(audio_fp)
    audio_fft = rfft(audio_fp)

    candidates = []

    # ── THE FIX: Clamp the search window to +/- 100 cents ──
    max_drift = 100
    # valid_indices represents 0 to +100 cents, and -100 to 0 cents (1100 to 1199)
    valid_indices = list(range(max_drift + 1)) + list(range(1200 - max_drift, 1200))

    for maqam_name, m in MAQAM.items():
        for upper_j in m.upper_options:
            scale = build_scale(maqam_name, upper_j)
            maqam_fp = _create_maqam_fingerprint(scale)
            maqam_fft = rfft(maqam_fp)
            cross_corr = irfft(audio_fft * np.conj(maqam_fft))

            # Only look for the highest correlation score within our realistic tuning bounds
            best_idx = valid_indices[np.argmax(cross_corr[valid_indices])]
            score = cross_corr[best_idx]

            offset_cents = best_idx
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

    if tonic_hz > 0.0 and tonic_confidence >= MAQAM_DETECTION['tonic_confidence_threshold']:
        for c in candidates:
            base_name = c.name.split(" (upper")[0].strip()
            if base_name not in MAQAM:
                continue
            candidate_tonic_hz = comma_to_freq(MAQAM[base_name].tonic_abs)
            dist_cents = abs(1200.0 * math.log2(tonic_hz / candidate_tonic_hz))
            if dist_cents > 600:
                dist_cents = abs(dist_cents - 1200)  # correct for octave error
            if dist_cents <= MAQAM_DETECTION['tonic_match_window_cents']:
                c.score *= MAQAM_DETECTION['tonic_match_boost']

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_n]


def extract_tuning_peaks_kde(
        frequencies: list[float],
        confidences: list[float],
        min_conf: float = None  # <--- THE FIX: Default to None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SOTA Dynamic Tuning using Kernel Density Estimation (KDE).
    """
    # Dynamically pull the exact threshold from config so it never starves!
    if min_conf is None:
        min_conf = PITCH_EXTRACTION['confidence_threshold']

    valid_idx = [i for i, (f, c) in enumerate(zip(frequencies, confidences)) if f > 0 and c > min_conf]
    if not valid_idx:
        return np.array([]), np.array([]), np.zeros(1200)

    valid_freqs = np.array([frequencies[i] for i in valid_idx])
    valid_confs = np.array([confidences[i] for i in valid_idx])

    cents = 1200 * np.log2(valid_freqs / 261.63)
    cents_folded = np.mod(cents, 1200)

    cents_padded = np.concatenate([cents_folded - 1200, cents_folded, cents_folded + 1200])
    weights_padded = np.concatenate([valid_confs, valid_confs, valid_confs])

    kde = gaussian_kde(cents_padded, weights=weights_padded, bw_method=MAQAM_DETECTION['kde_bandwidth'])

    x_grid = np.arange(1200)
    kde_curve = kde(x_grid)

    # --- ADD THIS NORMALIZATION FIX ---
    if np.max(kde_curve) > 0:
        kde_curve = kde_curve / np.max(kde_curve)

    peaks, _ = find_peaks(kde_curve, distance=35, prominence=0.05)

    return peaks, kde_curve[peaks], kde_curve

def detect_tonic(
        frequencies: list[float],
        confidences: list[float],
        hop_size_sec: float,
        min_conf: float = None,
) -> tuple[float, float]:
    """Returns (tonic_hz, tonic_confidence). Returns (0.0, 0.0) if detection fails."""
    if min_conf is None:
        min_conf = PITCH_EXTRACTION['confidence_threshold']

    C4 = 261.63

    voiced_idx = [i for i, (f, c) in enumerate(zip(frequencies, confidences))
                  if f > 0 and c > min_conf]
    if not voiced_idx:
        return 0.0, 0.0

    voiced_freqs = np.array([frequencies[i] for i in voiced_idx])
    voiced_confs = np.array([confidences[i] for i in voiced_idx])
    voiced_cents = 1200.0 * np.log2(voiced_freqs / C4)  # absolute, no folding

    evidence_cents = []
    evidence_weights = []
    evidence_confs = []

    voiced_mask = np.array([f > 0 and c > min_conf
                            for f, c in zip(frequencies, confidences)])

    # ── Evidence 1: Final held note (weight ×2.0) ──────────────────────
    # In Arabic taksim, the finishing note IS the tonic.  Use a short window
    # (~0.3 s) so we capture only the actual cadence note, not the melodic
    # movement leading up to it.
    final_window = max(3, int(0.3 / hop_size_sec))
    last_voiced = voiced_idx[-final_window:]
    if len(last_voiced) >= 3:
        pf = np.array([frequencies[i] for i in last_voiced])
        pc_ = np.array([confidences[i] for i in last_voiced])
        ev1_freq = float(np.median(pf))
        ev1_conf = float(np.mean(pc_))
        evidence_cents.append(1200.0 * math.log2(ev1_freq / C4))
        evidence_weights.append(2.0)
        evidence_confs.append(ev1_conf)

    # ── Evidence 2: Phrase-cadence notes (weight ×1.5) ─────────────────
    # Only count cadences followed by a real silence (≥150 ms), which
    # filters out ornamental micro-pauses between rapid notes.
    min_silence_frames = max(1, int(0.15 / hop_size_sec))
    phrase_end_freqs = []
    i = 0
    while i < len(voiced_mask) - 1:
        if voiced_mask[i] and not voiced_mask[i + 1]:
            # measure silence length
            silence_end = i + 1
            while silence_end < len(voiced_mask) and not voiced_mask[silence_end]:
                silence_end += 1
            if silence_end - i - 1 >= min_silence_frames:
                block = [frequencies[j] for j in range(max(0, i - 4), i + 1)
                         if voiced_mask[j]]
                if len(block) >= 2:
                    phrase_end_freqs.append(float(np.median(block)))
            i = silence_end
        else:
            i += 1
    if len(phrase_end_freqs) >= 3:
        pef_cents = 1200.0 * np.log2(np.array(phrase_end_freqs) / C4)
        ev2_cents = float(np.median(pef_cents))
        evidence_cents.append(ev2_cents)
        evidence_weights.append(1.5)
        evidence_confs.append(0.7)

    # ── Evidence 3: Histogram peak (weight ×0.3) ───────────────────────
    # Most-played note ≠ tonic (e.g. D is prominent in Rast on C), so
    # keep this as weak corroborating evidence only.
    cent_min = float(voiced_cents.min())
    cent_max = float(voiced_cents.max())
    span = cent_max - cent_min
    if span > 10:
        n_bins = max(50, int(span / 10))
        hist, bin_edges = np.histogram(voiced_cents, bins=n_bins,
                                       weights=voiced_confs, density=False)
        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=3.0)
        max_h = float(hist_smooth.max())
        if max_h > 0:
            peaks3, props3 = find_peaks(hist_smooth,
                                        distance=20,
                                        prominence=0.05 * max_h)
            if len(peaks3) > 0:
                top = peaks3[np.argmax(props3['prominences'])]
                bin_center = (bin_edges[top] + bin_edges[top + 1]) / 2.0
                prom_ratio = float(props3['prominences'].max()) / max_h
                evidence_cents.append(bin_center)
                evidence_weights.append(0.3)
                evidence_confs.append(prom_ratio)

    if not evidence_cents:
        return 0.0, 0.0

    # ── Combine: confidence×weight mean in cents space ──────────────────
    total_w = sum(w * c for w, c in zip(evidence_weights, evidence_confs))
    if total_w <= 0:
        return 0.0, 0.0

    combined_cents = sum(
        cents * w * c
        for cents, w, c in zip(evidence_cents, evidence_weights, evidence_confs)
    ) / total_w
    tonic_hz = C4 * (2.0 ** (combined_cents / 1200.0))
    overall_conf = float(np.mean(evidence_confs))

    logging.debug(f"[detect_tonic] tonic={tonic_hz:.1f}Hz conf={overall_conf:.3f} "
                  f"evidence_cents={[round(e, 1) for e in evidence_cents]}")
    return tonic_hz, overall_conf


def detect_maqam_with_consistency(
        frequencies: list[float],
        confidences: Optional[list[float]] = None,
        hop_size_sec: float = 0.01,
) -> list[MaqamCandidate]:
    """
    Main entry point for maqam detection.
    """
    return detect_maqam_sota(frequencies, confidences, top_n=5, hop_size_sec=hop_size_sec)
