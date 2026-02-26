"""
Maqam Auto-Detector

Given a sequence of detected frequencies, this module:
1. Builds a pitch histogram in 53-EDO comma space
2. Scores each candidate maqam by how well the histogram matches its scale
3. Returns a ranked list of maqam candidates with confidence scores

Key insight: we don't need perfect note-by-note detection to identify
the maqam. The statistical distribution of pitches in comma-space gives
a strong signal, especially because the microtonal degrees (Eb½, Bb½)
are diagnostic — their exact position distinguishes Rast from Bayati
from Sikah.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

from .tuning import (
    MAQAM,
    OCT,
    build_scale,
    ScaleNote,
)

# Keep your existing imports (MAQAM, OCT, build_scale, etc.)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

# How many cents wide is the "match window" around each scale degree?
# 50 cents = quarter tone — anything within this is considered a match
MATCH_WINDOW_CENTS = 35.0  # Was 50.0 - smaller means more precise
MATCH_WINDOW_COMMAS = MATCH_WINDOW_CENTS / (1200.0 / 53)  # ≈ 1.5 commas

# Weighting: microtonal degrees count more because they're more diagnostic
MICRO_WEIGHT = 2.5
NORMAL_WEIGHT = 1.0

# Tonic bonus: if the tonic pitch is strongly present, boost the score
TONIC_BONUS = 1.5

UPPER_JINS_BIAS = 1.15  # 15% boost for default upper jins

@dataclass
class MaqamCandidate:
    name: str
    score: float  # normalized 0..1
    tonic_detected: bool
    matched_degrees: list[str]
    scale: list[ScaleNote]

    def __repr__(self):
        return (f"MaqamCandidate({self.name!r}, score={self.score:.3f}, "
                f"tonic={self.tonic_detected}, matched={self.matched_degrees})")
def _score_maqam(
    histogram: np.ndarray,
    maqam_name: str,
    upper_jins: Optional[str] = None,
) -> MaqamCandidate:
    """Score a single maqam candidate against a pitch histogram."""
    m = MAQAM[maqam_name]
    scale = build_scale(maqam_name, upper_jins)

    score = 0.0
    max_possible = 0.0
    matched_degrees = []
    tonic_detected = False

    for note in scale:
        # Fold scale degree into one octave for histogram lookup
        rel = note.abs_commas % OCT
        if rel < 0:
            rel += OCT

        weight = MICRO_WEIGHT if note.is_micro else NORMAL_WEIGHT
        max_possible += weight

        # Sum histogram energy within match window around this scale degree
        energy = 0.0
        for bin_idx in range(53):
            bin_pos = float(bin_idx)
            dist = abs(rel - bin_pos)
            dist = min(dist, OCT - dist)
            if dist <= MATCH_WINDOW_COMMAS:
                # Cosine taper within window
                taper = math.cos(math.pi * dist / (2 * MATCH_WINDOW_COMMAS))
                energy += histogram[bin_idx] * taper

        degree_score = energy * weight
        score += degree_score

        if energy > 0.02:  # threshold to count as "matched"
            matched_degrees.append(note.label)
            if note.is_tonic:
                tonic_detected = True

    # Tonic bonus
    if tonic_detected:
        score *= TONIC_BONUS

    # Normalize
    normalized = score / (max_possible * TONIC_BONUS) if max_possible > 0 else 0.0
    normalized = min(normalized, 1.0)

    if upper_jins == m.upper_default:
        normalized *= UPPER_JINS_BIAS
        normalized = min(normalized, 1.0)

    if upper_jins != m.upper_default:
        display_name = f"{maqam_name} + upper {upper_jins}"
    else:
        display_name = maqam_name

    return MaqamCandidate(
        name=display_name,
        score=normalized,
        tonic_detected=tonic_detected,
        matched_degrees=matched_degrees,
        scale=scale,
    )



# ──────────────────────────────────────────────
# Pitch histogram builder
# ──────────────────────────────────────────────

def extract_tuning_peaks_kde(
        frequencies: list[float],
        confidences: list[float],
        min_conf: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SOTA Dynamic Tuning using Kernel Density Estimation (KDE).
    Finds the *actual* microtonal peaks played by the musician.
    """
    # 1. Filter out noise and unvoiced frames
    valid_idx = [i for i, (f, c) in enumerate(zip(frequencies, confidences)) if f > 0 and c > min_conf]
    if not valid_idx:
        return np.array([]), np.array([]), np.zeros(1200)

    valid_freqs = np.array([frequencies[i] for i in valid_idx])
    valid_confs = np.array([confidences[i] for i in valid_idx])

    # 2. Convert to Cents relative to C4 (261.63 Hz) and fold into 1 octave (0-1200 cents)
    cents = 1200 * np.log2(valid_freqs / 261.63)
    cents_folded = np.mod(cents, 1200)

    # 3. Handle Octave Wrap-Around for KDE
    # To ensure notes near C (0 or 1200) don't get cut off, we duplicate the data
    cents_padded = np.concatenate([cents_folded - 1200, cents_folded, cents_folded + 1200])
    weights_padded = np.concatenate([valid_confs, valid_confs, valid_confs])

    # 4. Calculate continuous Kernel Density Estimation (The "Hills")
    # Bandwidth of 0.02 roughly equals a ~24 cent smoothing window
    kde = gaussian_kde(cents_padded, weights=weights_padded, bw_method=0.02)

    # Evaluate the KDE over our 1-octave grid (0 to 1199 cents)
    x_grid = np.arange(0, 1200)
    kde_curve = kde(x_grid)

    # Normalize the curve
    kde_curve = kde_curve / np.max(kde_curve)

    # 5. Find the Peaks (The actual notes played)
    # distance=35 means notes must be at least 35 cents apart
    # prominence=0.05 filters out tiny noise bumps
    peaks, properties = find_peaks(kde_curve, distance=35, prominence=0.05)

    return peaks, kde_curve[peaks], kde_curve

# ──────────────────────────────────────────────
#  detector
# ──────────────────────────────────────────────

def detect_maqam_sota(
        frequencies: list[float],
        confidences: Optional[list[float]] = None,
        top_n: int = 3
) -> list[MaqamCandidate]:
    """
    Detects Maqam by comparing theoretical scales to the continuous KDE pitch curve.
    """
    if confidences is None:
        confidences = [1.0] * len(frequencies)

    # Generate the actual tuning fingerprint of the performer
    peaks, peak_heights, kde_curve = extract_tuning_peaks_kde(frequencies, confidences)

    candidates = []

    for maqam_name, m in MAQAM.items():
        for upper_j in m.upper_options:
            scale = build_scale(maqam_name, upper_j)

            score = 0.0
            max_possible = 0.0
            matched_degrees = []
            tonic_detected = False

            for note in scale:
                # Convert 53-EDO comma position to cents
                expected_cents = (note.abs_commas % 53) * (1200.0 / 53.0)

                # Weight microtonal notes heavier (they are the fingerprint of the maqam)
                weight = 2.5 if note.is_micro else 1.0
                max_possible += weight

                # Find the maximum KDE energy within a +/- 35 cent window of the expected note
                # This explicitly allows the performer to be "out of tune" with math, but in tune with tradition
                window_start = int(expected_cents - 35) % 1200
                window_end = int(expected_cents + 35) % 1200

                if window_start < window_end:
                    energy = np.max(kde_curve[window_start:window_end])
                else:  # Wrap around octave boundary
                    energy = max(np.max(kde_curve[window_start:]), np.max(kde_curve[:window_end]))

                degree_score = energy * weight
                score += degree_score

                if energy > 0.15:  # Significant presence
                    matched_degrees.append(note.label)
                    if note.is_tonic:
                        tonic_detected = True

            # Apply bonuses
            if tonic_detected: score *= 1.5
            if upper_j == m.upper_default: score *= 1.15

            normalized = min(score / (max_possible * 1.5), 1.0)

            name = f"{maqam_name} + upper {upper_j}" if upper_j != m.upper_default else maqam_name

            candidates.append(MaqamCandidate(
                name=name, score=normalized,
                tonic_detected=tonic_detected,
                matched_degrees=matched_degrees, scale=scale
            ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_n]


def detect_maqam_windowed(
        frequencies: list[float],
        confidences: Optional[list[float]] = None,
        frame_rate: float = 100.0,  # 100fps matches PENN's 10ms hopsize
        window_sec: float = 8.0,
) -> list[MaqamCandidate]:
    """
    Analyzes the opening phrase (Seyir) where the tonal center is usually established.
    """
    n_frames = int(window_sec * frame_rate)
    freqs = frequencies[:n_frames]
    confs = confidences[:n_frames] if confidences else None

    # Route to the new KDE engine
    return detect_maqam_sota(freqs, confs)


def detect_maqam_with_consistency(
        frequencies: list[float],
        confidences: Optional[list[float]] = None,
        frame_rate: float = 100.0,
) -> list[MaqamCandidate]:
    """
    Chunks the audio to find the 'home' Maqam, preventing temporary
    modulations in a Taksim from skewing the overall transcription.
    """
    segment_duration = 4.0  # seconds
    frames_per_segment = int(segment_duration * frame_rate)

    segment_candidates = []

    # 1. Chunk the performance and test each 4-second window
    for i in range(0, len(frequencies), frames_per_segment):
        seg_freqs = frequencies[i:i + frames_per_segment]
        seg_confs = confidences[i:i + frames_per_segment] if confidences else None

        if len(seg_freqs) < frames_per_segment // 2:
            continue

        candidates = detect_maqam_sota(seg_freqs, seg_confs, top_n=2)
        if candidates:
            segment_candidates.append(candidates[0].name)

    # 2. Find the 'home' maqam (most frequent across all chunks)
    from collections import Counter
    counter = Counter(segment_candidates)

    if counter:
        most_common = counter.most_common(1)[0][0]

        # 3. Re-score with full audio using the KDE engine, but bias toward the home maqam
        full_candidates = detect_maqam_sota(frequencies, confidences, top_n=5)
        for cand in full_candidates:
            if cand.name == most_common:
                cand.score *= 1.2  # Boost consistent 'home' maqam

        full_candidates.sort(key=lambda c: c.score, reverse=True)
        return full_candidates

    return detect_maqam_sota(frequencies, confidences)


def detect_modulations(
        frequencies: list[float],
        confidences: Optional[list[float]] = None,
        frame_rate: float = 100.0,
        window_sec: float = 5.0
) -> list[tuple[float, str]]:
    """
    Detects dynamic shifts (modulations) over time.
    Perfect for mapping a long Taksim or Wasla.
    """
    if confidences is None:
        confidences = [1.0] * len(frequencies)

    windows = []
    frame_window = int(window_sec * frame_rate)

    for i in range(0, len(frequencies), frame_window):
        window_freqs = frequencies[i:i + frame_window]
        window_confs = confidences[i:i + frame_window]

        # Only analyze windows with enough data
        if len(window_freqs) > frame_window / 2:
            candidates = detect_maqam_sota(window_freqs, window_confs, top_n=1)
            if candidates:
                time_in_seconds = i / frame_rate
                windows.append((time_in_seconds, candidates[0].name))

    return windows


# ─────────────────────
