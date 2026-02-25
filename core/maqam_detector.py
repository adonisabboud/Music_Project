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
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .tuning import (
    MAQAM,
    OCT,
    build_scale,
    freq_to_commas,
    ScaleNote,
)

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

def build_pitch_histogram(
        frequencies: list[float],
        confidences: Optional[list[float]] = None,
        min_freq: float = 80.0,
        max_freq: float = 1200.0,
) -> np.ndarray:
    """
    Build a 53-bin histogram of pitch energy in comma-space (one octave).

    Each detected frequency is folded into a single octave [0, 53)
    and accumulated into the nearest bin, weighted by confidence.

    Args:
        frequencies: List of detected frequencies in Hz (0 = unvoiced)
        confidences: Optional confidence weights per frame (0..1)
        min_freq: Ignore frequencies below this (likely noise)
        max_freq: Ignore frequencies above this

    Returns:
        np.ndarray of shape (53,) — normalized pitch histogram
    """
    histogram = np.zeros(53)

    if confidences is None:
        confidences = [1.0] * len(frequencies)

    for freq, conf in zip(frequencies, confidences):
        if freq <= min_freq or freq > max_freq or conf < 0.1:
            continue

        commas = freq_to_commas(freq)
        # Fold into one octave
        rel = commas % OCT
        if rel < 0:
            rel += OCT

        # Distribute weight into nearest bins (soft binning)
        bin_center = rel
        for bin_idx in range(53):
            bin_pos = float(bin_idx)
            dist = abs(bin_center - bin_pos)
            # Wrap-around distance
            dist = min(dist, OCT - dist)
            # Gaussian soft binning (sigma ≈ 1 comma)
            weight = conf * math.exp(-0.5 * (dist ** 2))
            histogram[bin_idx] += weight

    total = histogram.sum()
    if total > 0:
        histogram /= total

    return histogram


# ──────────────────────────────────────────────
#  detector
# ──────────────────────────────────────────────

def detect_maqam(
        frequencies: list[float],
        confidences: Optional[list[float]] = None,
        top_n: int = 3,
        min_freq: float = 80.0,
        max_freq: float = 1200.0,
) -> list[MaqamCandidate]:
    """
    Auto-detect the maqam from a list of detected frequencies.

    Args:
        frequencies: Per-frame fundamental frequencies in Hz (0 = unvoiced)
        confidences: Per-frame confidence values (0..1)
        top_n: How many top candidates to return
        min_freq: Minimum plausible frequency
        max_freq: Maximum plausible frequency

    Returns:
        List of MaqamCandidate objects, sorted by score descending.
        The first entry is the most likely maqam.
    """
    histogram = build_pitch_histogram(frequencies, confidences, min_freq, max_freq)

    candidates = []
    for maqam_name, m in MAQAM.items():
        # Score with each possible upper jins
        for upper_j in m.upper_options:
            c = _score_maqam(histogram, maqam_name, upper_j)
            # Label with upper jins if different from default
            if upper_j != m.upper_default:
                c.name = f"{maqam_name} + upper {upper_j}"
            candidates.append(c)

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_n]


def detect_maqam_windowed(
        frequencies: list[float],
        confidences: Optional[list[float]] = None,
        frame_rate: float = 100.0,  # frames per second (CREPE default)
        window_sec: float = 8.0,  # analyze first N seconds for maqam ID
) -> list[MaqamCandidate]:
    """
    Use only the first `window_sec` of audio for maqam detection.
    The opening of a piece is usually the most tonally clear.
    """
    n_frames = int(window_sec * frame_rate)
    freqs = frequencies[:n_frames]
    confs = confidences[:n_frames] if confidences else None
    return detect_maqam(freqs, confs)


def detect_maqam_with_consistency(
        frequencies: list[float],
        confidences: Optional[list[float]] = None,
        frame_rate: float = 100.0,
) -> list[MaqamCandidate]:
    """
    Detect maqam with temporal consistency checks.
    """
    # Split into segments and detect maqam per segment
    segment_duration = 4.0  # seconds
    frames_per_segment = int(segment_duration * frame_rate)

    segment_candidates = []
    for i in range(0, len(frequencies), frames_per_segment):
        seg_freqs = frequencies[i:i + frames_per_segment]
        seg_confs = confidences[i:i + frames_per_segment] if confidences else None

        if len(seg_freqs) < frames_per_segment // 2:
            continue

        candidates = detect_maqam(seg_freqs, seg_confs, top_n=2)
        if candidates:
            segment_candidates.append(candidates[0].name)

    # Find most consistent maqam across segments
    from collections import Counter
    counter = Counter(segment_candidates)
    if counter:
        most_common = counter.most_common(1)[0][0]

        # Re-score with full audio but bias toward consistent maqam
        full_candidates = detect_maqam(frequencies, confidences, top_n=5)
        for cand in full_candidates:
            if cand.name == most_common:
                cand.score *= 1.2  # Boost consistent maqam

        full_candidates.sort(key=lambda c: c.score, reverse=True)
        return full_candidates

    return detect_maqam(frequencies, confidences)

def detect_modulations(
        frequencies: list[float],
        confidences: Optional[list[float]] = None,
        frame_rate: float = 100.0,
        window_sec: float = 5.0
) -> list[tuple[float, str]]:
    """
    Detect if the maqam changes over time by analyzing sliding windows.

    Args:
        frequencies: Per-frame fundamental frequencies in Hz
        confidences: Per-frame confidence values
        frame_rate: Frames per second
        window_sec: Duration of each analysis window in seconds

    Returns:
        List of (time_in_seconds, maqam_name) tuples for each window
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
            candidates = detect_maqam(window_freqs, window_confs, top_n=1)
            if candidates:
                time_in_seconds = i / frame_rate
                windows.append((time_in_seconds, candidates[0].name))

    return windows


# ─────────────────────

# ──────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Maqam Detector Self-Test ===\n")

    # Simulate a Bayati on D performance by generating frequencies
    # from the Bayati scale with some noise and ornamentation
    rng = np.random.default_rng(42)

    bayati_scale = build_scale("Bayati on D")
    test_freqs = []
    test_confs = []

    for _ in range(500):
        note = rng.choice(bayati_scale)
        # Add slight pitch variation (±15 cents) to simulate real performance
        variation_cents = rng.normal(0, 15)
        variation_factor = 2 ** (variation_cents / 1200)
        freq = note.freq_hz * variation_factor
        test_freqs.append(freq)
        test_confs.append(rng.uniform(0.7, 1.0))

    # Add some silence frames
    test_freqs += [0.0] * 50
    test_confs += [0.0] * 50

    print("Simulated: Bayati on D with ±15 cent variation + 10% silence\n")
    results = detect_maqam(test_freqs, test_confs)

    for i, c in enumerate(results):
        print(f"  #{i + 1}: {c.name}")
        print(f"        score={c.score:.4f}, tonic_detected={c.tonic_detected}")
        print(f"        matched: {c.matched_degrees}")
        print()

    # Test Rast
    print("─" * 50)
    rast_scale = build_scale("Rast on C")
    test_freqs2 = []
    test_confs2 = []
    for _ in range(500):
        note = rng.choice(rast_scale)
        variation_cents = rng.normal(0, 15)
        freq = note.freq_hz * (2 ** (variation_cents / 1200))
        test_freqs2.append(freq)
        test_confs2.append(rng.uniform(0.7, 1.0))

    print("\nSimulated: Rast on C with ±15 cent variation\n")
    results2 = detect_maqam(test_freqs2, test_confs2)
    for i, c in enumerate(results2):
        print(f"  #{i + 1}: {c.name}")
        print(f"        score={c.score:.4f}, tonic_detected={c.tonic_detected}")
        print()
