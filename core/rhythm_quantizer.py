"""
Rhythm Quantizer with Arabic Iqa'at Support
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


# Standard rhythmic values as fractions of a whole note
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
    Fraction(1, 24),   # triplet sixteenth
    Fraction(1, 32),   # thirty-second
]

BPM_CANDIDATES = list(range(40, 200, 2))

# Iqa'at (Arabic rhythmic cycles)
try:
    from .tuning import IQA_AT
except ImportError:
    IQA_AT = {
        "taksim": {"name": "Taksim (Free Rhythm)", "type": "free", "pattern": []},
        "maqsum": {"name": "Maqsum", "type": "metered", "pattern": [2, 2, 2, 2, 2, 2, 2, 2]},
        "wahda": {"name": "Wahda", "type": "metered", "pattern": [4, 4]},
        "sama'i": {"name": "Sama'i Thaqil", "type": "metered", "pattern": [3, 3, 3, 4]},
    }


def _nearest_rhythmic_value(beats: float, prefer_small: bool = True) -> Fraction:
    """Find the nearest standard rhythmic value, preferring smaller values for ornaments."""
    best = Fraction(1, 16)  # Default to sixteenth
    best_dist = float("inf")

    for val in RHYTHMIC_VALUES:
        dist = abs(beats - float(val))
        # If distances are very close, prefer the smaller value
        if abs(dist - best_dist) < 0.05:
            if prefer_small and val < best:
                best = val
                best_dist = dist
        elif dist < best_dist:
            best_dist = dist
            best = val

    return best  # Changed from 'bestest' to 'best'

def estimate_tempo(note_segments: List[dict], default_bpm: float = 80.0) -> float:
    """Estimate tempo from note durations."""
    durations = np.array([n["duration_sec"] for n in note_segments if n["duration_sec"] > 0.05])

    if len(durations) < 4:
        return default_bpm

    best_bpm = default_bpm
    best_score = float("inf")

    for bpm in BPM_CANDIDATES:
        beat_dur = 60.0 / bpm
        total_error = 0.0
        for dur in durations:
            beats = dur / beat_dur
            nearest = _nearest_rhythmic_value(beats)
            error = abs(beats - float(nearest))
            total_error += error

        if total_error < best_score:
            best_score = total_error
            best_bpm = bpm

    return float(best_bpm)


def estimate_tempo_arabic(note_segments: List[dict]) -> float:
    """Estimate tempo with awareness of Arabic rhythmic patterns."""
    durations = np.array([n["duration_sec"] for n in note_segments if n["duration_sec"] > 0.05])

    if len(durations) < 4:
        return 80.0

    sorted_durs = np.sort(durations)
    main_note_durs = sorted_durs[len(sorted_durs)//4: 3*len(sorted_durs)//4]

    if len(main_note_durs) < 2:
        main_note_durs = sorted_durs

    beat_candidates = []
    for dur in main_note_durs:
        if 0.3 < dur < 1.5:
            beat_candidates.append(dur)

    if beat_candidates:
        median_beat = np.median(beat_candidates)
        bpm = 60.0 / median_beat
        return round(bpm)

    return 80.0


def detect_iqa(note_segments: List[dict], bpm: float) -> Optional[str]:
    """Detect the most likely iqa' from note patterns."""
    if not RHYTHM_QUANTIZATION['iqa_detection_enabled'] or len(note_segments) < 8:
        return None

    beat_dur = 60.0 / bpm
    durations = [n["duration_sec"] / beat_dur for n in note_segments]

    best_match = None
    best_score = float("inf")

    for iqa_name, iqa in IQA_AT.items():
        if iqa.get("type") != "metered" or not iqa.get("pattern"):
            continue

        pattern = iqa["pattern"]
        for start in range(len(durations) - len(pattern)):
            segment = durations[start:start + len(pattern)]
            min_dur = min(segment)
            if min_dur == 0:
                continue
            norm_durs = [d / min_dur for d in segment]
            pattern_norm = [p / min(pattern) for p in pattern]
            error = sum(abs(n - p) for n, p in zip(norm_durs, pattern_norm)) / len(pattern)

            if error < best_score and error < 0.5:
                best_score = error
                best_match = iqa_name

    return best_match


def quantize_rhythm(
    note_segments: List[dict],
    quantized_pitches: List[Optional[object]],
    bpm: Optional[float] = None,
) -> Tuple[List[QuantizedNote], float]:
    """Basic rhythm quantization."""
    if bpm is None:
        bpm = estimate_tempo(note_segments)

    beat_dur = 60.0 / bpm
    quantized_notes = []
    current_beat = 0.0

    for seg, pitch in zip(note_segments, quantized_pitches):
        if pitch is None:
            current_beat += seg["duration_sec"] / beat_dur
            continue

        dur_beats = seg["duration_sec"] / beat_dur
        quantized_dur = _nearest_rhythmic_value(dur_beats)

        if quantized_dur < Fraction(1, 16):
            quantized_dur = Fraction(1, 16)

        qn = QuantizedNote(
            start_beat=current_beat,
            duration_beats=quantized_dur,
            duration_sec=seg["duration_sec"],
            median_freq=seg["median_freq"],
            note_label=pitch.label,
            is_micro=pitch.is_micro,
            is_ornate=seg.get("is_ornate", False),
            mean_confidence=seg.get("mean_confidence", 1.0),
        )
        quantized_notes.append(qn)
        current_beat += float(quantized_dur)

    return quantized_notes, bpm


def _quantize_with_iqa_context(
    actual_beats: float,
    expected_beats: float,
    confidence: float,
) -> Fraction:
    """Quantize duration considering iqa' context."""
    if confidence > 0.8:
        nearest = _nearest_rhythmic_value(actual_beats)
        expected_frac = Fraction(expected_beats).limit_denominator(32)
        if abs(actual_beats - expected_beats) < 0.2:
            return expected_frac
        return nearest
    return Fraction(expected_beats).limit_denominator(32)




def quantize_rhythm_arabic(
        note_segments: List[dict],
        quantized_pitches: List[Optional[object]],
        bpm: Optional[float] = None,
        iqa_override: Optional[str] = None,
) -> Tuple[List[QuantizedNote], float, Optional[str]]:
    """Quantize rhythm with awareness of fast ornamental passages."""

    if bpm is None:
        bpm = estimate_tempo_arabic(note_segments)

    detected_iqa = iqa_override or detect_iqa(note_segments, bpm)

    beat_dur = 60.0 / bpm
    quantized_notes = []
    current_beat = 0.0

    # First pass: detect if this is a fast ornamental passage
    all_durations = [s["duration_sec"] for s in note_segments]
    avg_duration = np.mean(all_durations)
    median_duration = np.median(all_durations)

    # If average note is very short (< 150ms), we're in a fast passage
    is_fast_passage = median_duration < 0.15

    if is_fast_passage:
        # For fast passages, use smaller rhythmic values
        # Don't merge notes - keep them separate!
        min_allowed_duration = Fraction(1, 32)  # 32nd notes
    else:
        min_allowed_duration = Fraction(1, 16)  # 16th notes

    # If iqa' detected, use its structure
    if detected_iqa and detected_iqa in IQA_AT and IQA_AT[detected_iqa].get("type") == "metered":
        iqa = IQA_AT[detected_iqa]
        pattern = iqa["pattern"]
        pattern_len = len(pattern)
        pattern_beat_sum = sum(pattern)

        for i, (seg, pitch) in enumerate(zip(note_segments, quantized_pitches)):
            if pitch is None:
                current_beat += seg["duration_sec"] / beat_dur
                continue

            pos_in_cycle = i % pattern_len
            expected_dur = pattern[pos_in_cycle] / (pattern_beat_sum / 4)

            dur_beats = seg["duration_sec"] / beat_dur
            quantized_dur = _quantize_with_iqa_context(
                dur_beats, expected_dur, seg.get("mean_confidence", 1.0)
            )

            # Ensure we don't exceed minimum duration in fast passages
            if quantized_dur < min_allowed_duration:
                quantized_dur = min_allowed_duration

            qn = QuantizedNote(
                start_beat=current_beat,
                duration_beats=quantized_dur,
                duration_sec=seg["duration_sec"],
                median_freq=seg["median_freq"],
                note_label=pitch.label,
                is_micro=pitch.is_micro,
                is_ornate=seg.get("is_ornate", False),
                mean_confidence=seg.get("mean_confidence", 1.0),
                iqa_position=pos_in_cycle if detected_iqa else None,
            )
            quantized_notes.append(qn)
            current_beat += float(quantized_dur)

    else:
        # No iqa' - use basic quantization but with fast passage awareness
        for seg, pitch in zip(note_segments, quantized_pitches):
            if pitch is None:
                current_beat += seg["duration_sec"] / beat_dur
                continue

            dur_beats = seg["duration_sec"] / beat_dur
            quantized_dur = _nearest_rhythmic_value(dur_beats)

            # In fast passages, never merge notes into longer values
            if is_fast_passage:
                # Force small note values for fast passages
                if quantized_dur > Fraction(1, 8):  # If trying to make quarter or longer
                    # Break into eighth notes
                    num_notes = int(float(quantized_dur) / 0.5)  # 0.5 = eighth note in beats
                    for j in range(num_notes):
                        qn = QuantizedNote(
                            start_beat=current_beat + (j * 0.5),
                            duration_beats=Fraction(1, 8),
                            duration_sec=seg["duration_sec"] / num_notes,
                            median_freq=seg["median_freq"],
                            note_label=pitch.label,
                            is_micro=pitch.is_micro,
                            is_ornate=seg.get("is_ornate", False),
                            mean_confidence=seg.get("mean_confidence", 1.0),
                        )
                        quantized_notes.append(qn)
                    current_beat += float(quantized_dur)
                    continue

            # Normal case
            if quantized_dur < min_allowed_duration:
                quantized_dur = min_allowed_duration

            qn = QuantizedNote(
                start_beat=current_beat,
                duration_beats=quantized_dur,
                duration_sec=seg["duration_sec"],
                median_freq=seg["median_freq"],
                note_label=pitch.label,
                is_micro=pitch.is_micro,
                is_ornate=seg.get("is_ornate", False),
                mean_confidence=seg.get("mean_confidence", 1.0),
            )
            quantized_notes.append(qn)
            current_beat += float(quantized_dur)

    return quantized_notes, bpm, detected_iqa


def quantize_rhythm_taksim(
        note_segments: List[dict],
        quantized_pitches: List[Optional[object]],
) -> Tuple[List[QuantizedNote], float, Optional[str]]:
    """
    Taksim mode MVP: Maps raw durations to the closest standard sheet music fraction
    so the XML exporter can read it safely without dropping notes.
    """
    quantized_notes = []
    current_beat = 0.0

    # Let's use 60 BPM for Taksim so 1 beat = 1 second.
    # This makes the math incredibly clean: duration_sec == duration_beats.
    reference_bpm = 60.0

    for seg, pitch in zip(note_segments, quantized_pitches):
        if pitch is None:
            # Still advance the time tracker for rests!
            # Since 60 BPM = 1 beat/sec, duration_sec is exactly the number of beats
            current_beat += seg["duration_sec"]
            continue

        # In standard notation, a whole note is 4 beats.
        # We divide by 4.0 to convert our beat count into a whole-note fraction for XML.
        beats_for_xml = seg["duration_sec"] / 4.0

        # Snap it to a safe fraction that your MusicXML exporter understands
        quantized_dur = _nearest_rhythmic_value(beats_for_xml)

        qn = QuantizedNote(
            start_beat=current_beat,
            duration_beats=quantized_dur,
            duration_sec=seg["duration_sec"],
            median_freq=seg["median_freq"],
            note_label=pitch.label,
            is_micro=pitch.is_micro,
            is_ornate=False,  # Forced False for the MVP
            mean_confidence=seg.get("mean_confidence", 1.0),
        )
        quantized_notes.append(qn)

        # Advance the beat tracker.
        # We multiply the fraction by 4.0 to convert it back to standard quarter-note beats.
        # (e.g., Fraction(1, 4) * 4.0 = 1.0 standard beat)
        current_beat += float(quantized_dur) * 4.0

    return quantized_notes, reference_bpm, "taksim"


def notes_to_measures(
    notes: List[QuantizedNote],
    time_signature: tuple[int, int] = (4, 4),
) -> List[List[QuantizedNote]]:
    """Group quantized notes into measures."""
    beats_per_measure = time_signature[0] * (4 / time_signature[1])
    measures = []
    current_measure = []
    measure_start_beat = 0.0

    for note in notes:
        if note.start_beat >= measure_start_beat + beats_per_measure:
            if current_measure:
                measures.append(current_measure)
            measures_skipped = int(
                (note.start_beat - measure_start_beat) / beats_per_measure
            )
            measure_start_beat += measures_skipped * beats_per_measure
            current_measure = []
        current_measure.append(note)

    if current_measure:
        measures.append(current_measure)

    return measures