"""
SOTA Rhythm Quantizer with Arabic Iqa'at and Rubato/Taksim Support
"""

from fractions import Fraction
from typing import Optional, Tuple, List
import numpy as np
from dataclasses import dataclass, field

from .config import RHYTHM_QUANTIZATION

# ──────────────────────────────────────────────
# Iqa'at (Arabic Rhythmic Cycles)
# ──────────────────────────────────────────────

IQA_AT = {
    "taksim": {
        "name": "Taksim (Free Rhythm)",
        "type": "free",  # Special type for unmetered sections
        "description": "Improvisational, non-metric section",
        "beats": None,  # No fixed beats
        "time_signature": None,  # No fixed time signature
        "pattern": [],  # No rhythmic pattern
        "typical_duration": "variable",  # Can be any length
    },
    "maqsum": {
        "name": "Maqsum",
        "type": "metered",
        "pattern": [2, 2, 2, 2, 2, 2, 2, 2],  # 4/4 with specific accents
        "beats": 8,
        "time_signature": (4, 4),
        "dum_tek": ["D", "T", "D", "T", "T", "D", "T", "T"],
        "description": "Most common 4/4 rhythm",
    },
    "wahda": {
        "name": "Wahda",
        "type": "metered",
        "pattern": [4, 4],  # 4/4
        "beats": 8,
        "time_signature": (4, 4),
        "description": "Simple 4/4 rhythm",
    },
    "sama'i": {
        "name": "Sama'i Thaqil",
        "type": "metered",
        "pattern": [3, 3, 3, 4],  # 10/8
        "beats": 10,
        "time_signature": (10, 8),
        "description": "10/8 rhythm common in instrumental forms",
    },
    "masmoudi": {
        "name": "Masmoudi Kabir",
        "type": "metered",
        "pattern": [2, 2, 2, 2, 2, 2],  # 4/4
        "beats": 8,
        "time_signature": (4, 4),
        "description": "Heavy 4/4 rhythm",
    },
    "ayoub": {
        "name": "Ayoub",
        "type": "metered",
        "pattern": [3, 3],  # 3/4
        "beats": 6,
        "time_signature": (3, 4),
        "description": "3/4 rhythm often used in religious music",
    },
    "darij": {
        "name": "Darij",
        "type": "metered",
        "pattern": [3, 2, 2],  # 7/8
        "beats": 7,
        "time_signature": (7, 8),
        "description": "7/8 rhythm common in folk music",
    },
    "malfuf": {
        "name": "Malfuf",
        "type": "metered",
        "pattern": [2, 2, 2, 2],  # 2/4
        "beats": 4,
        "time_signature": (2, 4),
        "description": "Fast 2/4 rhythm",
    },
    "jerk": {
        "name": "Jerk",
        "type": "metered",
        "pattern": [3, 3, 2],  # 8/8
        "beats": 8,
        "time_signature": (8, 8),
        "description": "8/8 rhythm with asymmetrical pattern",
    },
    "samai_darij": {
        "name": "Sama'i Darij",
        "type": "metered",
        "pattern": [3, 3, 4],  # 10/8 (alternative pattern)
        "beats": 10,
        "time_signature": (10, 8),
        "description": "10/8 rhythm variant",
    },
}


def detect_rhythm_type(segments: list) -> str:
    """
    Detect if a section is taksim (free rhythm) or metered.
    Useful for separating improvisational sections.
    """
    if len(segments) < 10:
        return "taksim"  # Short sections are often taksim

    # Calculate rhythm regularity
    durations = [s["duration_sec"] for s in segments if s["duration_sec"] > 0.05]
    if len(durations) < 5:
        return "taksim"

    # Check for repetitive patterns
    cv = np.std(durations) / np.mean(durations)  # Coefficient of variation
    if cv > 0.8:  # High variation indicates free rhythm
        return "taksim"

    # Try to detect specific iqa'at patterns
    # Normalize durations to find common patterns
    min_dur = min(durations)
    norm_durs = [round(d / min_dur) for d in durations]

    # Look for repeating patterns of length 4-8
    for pattern_len in [4, 6, 8, 10]:
        if len(norm_durs) >= pattern_len * 2:
            pattern = norm_durs[:pattern_len]
            # Check if pattern repeats
            next_pattern = norm_durs[pattern_len:pattern_len * 2]
            if pattern == next_pattern:
                # Found repeating pattern
                pattern_sum = sum(pattern)
                for iqa_name, iqa in IQA_AT.items():
                    if iqa["type"] == "metered" and iqa.get("pattern"):
                        if len(iqa["pattern"]) == pattern_len and sum(iqa["pattern"]) == pattern_sum:
                            # Check if patterns match closely
                            match = all(abs(p1 - p2) <= 1 for p1, p2 in zip(pattern, iqa["pattern"]))
                            if match:
                                return iqa_name
                return "metered_unknown"

    return "taksim" if cv > 0.6 else "metered_unknown"


@dataclass
class QuantizedNote:
    start_beat: float
    duration_beats: Fraction
    duration_sec: float
    median_freq: float
    note_label: str
    is_micro: bool
    is_ornate: bool
    mean_confidence: float
    ornaments_before: list = field(default_factory=list)
    ornaments_after: list = field(default_factory=list)
    iqa_position: Optional[int] = None

# Streamlined rhythmic values for cleaner sheet music
RHYTHMIC_VALUES = [
    Fraction(1, 1),    # whole
    Fraction(3, 4),    # dotted half
    Fraction(1, 2),    # half
    Fraction(3, 8),    # dotted quarter
    Fraction(1, 4),    # quarter
    Fraction(3, 16),   # dotted eighth
    Fraction(1, 8),    # eighth
    Fraction(1, 12),   # triplet eighth
    Fraction(1, 16),   # sixteenth
]

try:
    from .tuning import IQA_AT
except ImportError:
    IQA_AT = {
        "taksim": {"name": "Taksim (Free Rhythm)", "type": "free", "pattern": []},
        "maqsum": {"name": "Maqsum", "type": "metered", "pattern": [2, 2, 2, 2, 2, 2, 2, 2]},
    }

def _nearest_rhythmic_value(beats: float, prefer_small: bool = True) -> Fraction:
    """Find the nearest clean standard rhythmic value."""
    best = Fraction(1, 16)
    best_dist = float("inf")

    for val in RHYTHMIC_VALUES:
        dist = abs(beats - float(val))
        if abs(dist - best_dist) < 0.05:
            if prefer_small and val < best:
                best = val
                best_dist = dist
        elif dist < best_dist:
            best_dist = dist
            best = val

    return best

def _clean_legato_and_ornaments(note_segments: List[dict], quantized_pitches: List[Optional[object]]) -> List[dict]:
    """
    SOTA Pre-processing:
    1. Fills micro-rests (legato smoothing)
    2. Identifies grace notes (fast ornaments)
    """
    cleaned = []

    for i, (seg, pitch) in enumerate(zip(note_segments, quantized_pitches)):
        if pitch is None:
            continue

        dur = seg["duration_sec"]

        # 1. Grace Note Detection (< 90ms is usually a fast ornament/pluck)
        is_ornament = dur < 0.09

        # 2. Legato Gap Filling
        # Look ahead to the next note. If the gap is tiny (< 0.05s),
        # extend this note's duration to touch the next one.
        if i < len(note_segments) - 1:
            next_seg = note_segments[i+1]
            gap = next_seg["start_time"] - (seg["start_time"] + dur)
            if 0 < gap < 0.05:
                dur += gap # Swallow the rest!

        cleaned.append({
            "start_time": seg["start_time"],
            "duration_sec": dur,
            "median_freq": seg["median_freq"],
            "pitch_obj": pitch,
            "mean_confidence": seg.get("mean_confidence", 1.0),
            "is_ornate": is_ornament
        })

    return cleaned

def quantize_rhythm_taksim(
        note_segments: List[dict],
        quantized_pitches: List[Optional[object]],
) -> Tuple[List[QuantizedNote], float, Optional[str]]:
    """
    SOTA Taksim Quantizer:
    Anchors to a floating 'local pulse' and extracts fast ornaments
    as grace notes for beautiful, readable sheet music.
    """
    # 1. Clean the raw data (remove micro-rests, flag ornaments)
    cleaned_data = _clean_legato_and_ornaments(note_segments, quantized_pitches)

    if not cleaned_data:
        return [], 60.0, "taksim"

    # 2. Establish the 'Floating Pulse'
    # Find the median duration of the main structural notes (ignoring ornaments)
    main_notes = [n["duration_sec"] for n in cleaned_data if not n["is_ornate"]]

    # If the whole phrase is just fast notes, fallback to a standard pulse
    median_pulse_sec = np.median(main_notes) if main_notes else 0.5

    # We will treat this median pulse as an 8th note.
    # Therefore, 1 Quarter Note (1 beat) = median_pulse_sec * 2
    beat_dur_sec = median_pulse_sec * 2.0
    reference_bpm = 60.0 / beat_dur_sec

    # Cap BPM to sane limits for notation software
    reference_bpm = max(40.0, min(reference_bpm, 120.0))
    beat_dur_sec = 60.0 / reference_bpm

    quantized_notes = []
    current_beat = 0.0
    pending_ornaments = []

    for item in cleaned_data:
        pitch = item["pitch_obj"]

        if item["is_ornate"]:
            # Keep grace notes out of the main time grid
            qn = QuantizedNote(
                start_beat=current_beat,
                duration_beats=Fraction(0, 1), # Takes 0 logical time in XML
                duration_sec=item["duration_sec"],
                median_freq=item["median_freq"],
                note_label=pitch.label,
                is_micro=pitch.is_micro,
                is_ornate=True,
                mean_confidence=item["mean_confidence"],
            )
            pending_ornaments.append(qn)
            continue

        # Process main structural notes
        beats_for_xml = item["duration_sec"] / beat_dur_sec
        quantized_dur = _nearest_rhythmic_value(beats_for_xml / 4.0) # Convert to whole-note fraction

        qn = QuantizedNote(
            start_beat=current_beat,
            duration_beats=quantized_dur,
            duration_sec=item["duration_sec"],
            median_freq=item["median_freq"],
            note_label=pitch.label,
            is_micro=pitch.is_micro,
            is_ornate=False,
            mean_confidence=item["mean_confidence"],
            ornaments_before=pending_ornaments.copy()
        )
        quantized_notes.append(qn)

        # Advance the beat tracker
        current_beat += float(quantized_dur) * 4.0
        pending_ornaments = [] # clear them since they are attached

    return quantized_notes, reference_bpm, "taksim"

def estimate_tempo(note_segments: List[dict]) -> float:
    """Placeholder for tempo estimation."""
    return 120.0

def detect_iqa(note_segments: List[dict], tempo: float) -> str:
    """Placeholder for Iqa detection."""
    return "maqsum"

def notes_to_measures(quantized_notes: List[QuantizedNote], time_signature: Tuple[int, int] = (4, 4)) -> List[List[QuantizedNote]]:
    """
    Groups quantized notes into measures based on the time signature.
    """
    if not quantized_notes:
        return []

    numerator, denominator = time_signature
    measure_duration_beats = Fraction(numerator, denominator) * 4 # in quarter notes

    measures = []
    current_measure = []
    current_measure_beats = Fraction(0)

    for note in quantized_notes:
        # note.duration_beats is in whole notes, so multiply by 4 to get quarter notes
        note_duration_beats = note.duration_beats * 4

        if current_measure_beats + note_duration_beats > measure_duration_beats:
            if current_measure:
                measures.append(current_measure)
            
            current_measure = [note]
            current_measure_beats = note_duration_beats
        else:
            current_measure.append(note)
            current_measure_beats += note_duration_beats

        if current_measure_beats == measure_duration_beats:
            measures.append(current_measure)
            current_measure = []
            current_measure_beats = Fraction(0)

    if current_measure:
        measures.append(current_measure)

    return measures
