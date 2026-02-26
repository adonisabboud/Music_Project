"""
SOTA Rhythm Quantizer with Arabic Iqa'at and Rubato/Taksim Support
"""

from fractions import Fraction
from typing import Optional, Tuple, List
import numpy as np
from dataclasses import dataclass, field

from .config import RHYTHM_QUANTIZATION

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

# ... (Keep estimate_tempo, detect_iqa, quantize_rhythm_arabic, and notes_to_measures the same as your previous version, as they work well for rigid Iqa'at) ...