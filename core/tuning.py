"""
53-EDO Turkish Comma Tuning Engine with Arabic Iqa'at Support
Port of maqam-builder-53edo/app.js tuning logic to Python.

All musical logic operates in comma-space.
Frequencies are calculated only when needed.
"""

import math
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import copy
# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
C4 = 261.63  # Standard concert pitch instead of 260.77
COMMA = 2 ** (1/53)  # One Turkish comma
OCT = 53             # Commas per octave

# Pitch classes in comma-space (relative to C4)
PC = {
    "C":  0,
    "D":  9,
    "E":  18,
    "F":  22,
    "G":  31,
    "A":  40,
    "B":  49,
    "C5": 53,
}

# Chromatic / diatonic semitone sizes
CHROM = 5  # chromatic semitone (e.g. C→C#)
DIAT  = 4  # diatonic semitone  (e.g. E→F)

# Microtonal notes (in comma-space, relative to C4)
def _mid(a, b):
    return (a + b) / 2.0

MICRO = {
    "Eb_half_rast":    _mid(PC["D"], PC["F"]),          # 15.5
    "Eb_half_bayati":  _mid(PC["D"], PC["F"]) - 0.5,   # 15.0
    "Eb_half_sikah":   _mid(PC["D"], PC["F"]) + 0.5,   # 16.0

    "Bb_half_rast":    _mid(PC["A"], PC["C5"]),         # 44.5  (note: C5=53)
    "Bb_half_bayati":  _mid(PC["A"], PC["C5"]) - 0.5,  # 44.0
    "Bb_half_sikah":   _mid(PC["A"], PC["C5"]) + 0.5,  # 45.0
    
    # Additional microtonal positions
    "F_half_sharp": _mid(PC["F"], PC["F"] + CHROM),    # 24.5
    "C_half_sharp": _mid(PC["C"], PC["C"] + CHROM),    # 2.5
    "G_half_sharp": _mid(PC["G"], PC["G"] + CHROM),    # 33.5
    "A_half_flat": _mid(PC["A"], PC["A"] - DIAT),      # 38.0
    "B_half_flat": _mid(PC["B"], PC["B"] - DIAT),      # 47.0
    
    # Specific maqam inflections
    "Saba_F_flat": PC["F"] - 1.0,                       # 21.0
    "Hijaz_E_quarter": PC["D"] + 15.0,                  # 24.0
}


# ──────────────────────────────────────────────
# Frequency ↔ Comma conversion
# ──────────────────────────────────────────────

def comma_to_freq(commas_from_c4: float, reference_c4: float = C4) -> float:
    """Convert comma position (relative to C4) to frequency in Hz."""
    return reference_c4 * (COMMA ** commas_from_c4)

def freq_to_commas(freq_hz: float, reference_c4: float = C4) -> float:
    """Convert frequency in Hz to comma position relative to C4.
    This is the inverse of comma_to_freq."""
    return math.log2(freq_hz / reference_c4) * 53

def commas_to_octave_relative(commas: float) -> float:
    """Reduce a comma value to within one octave [0, 53)."""
    return commas % OCT

# ──────────────────────────────────────────────
# Jins definitions
# ──────────────────────────────────────────────

@dataclass
class Jins:
    name: str
    offsets: list[float]   # comma offsets from base note
    labels: list[str]

JINS = {
    "Rast":     Jins("Rast",     [0, 9, 15.5, 22],  ["T", "2", "Eb½ (Rast)", "4"]),
    "Bayati":   Jins("Bayati",   [0, 6.0, 13, 22],  ["T", "Eb½ (Bayati)", "F", "G"]),
    "Sikah":    Jins("Sikah",    [0, 6, 15],         ["T (Eb½ Sikah)", "F", "G"]),
    "Hijaz":    Jins("Hijaz",    [0, 4, 18, 22],     ["T", "Eb", "F#", "G"]),
    "Kurd":     Jins("Kurd",     [0, 4, 13, 22],     ["T", "Eb", "F", "G"]),
    "Nahawand": Jins("Nahawand", [0, 9, 13, 22],     ["T", "D", "Eb", "F"]),
    "Ajam":     Jins("Ajam",     [0, 9, 18, 22],     ["T", "D", "E", "F"]),
    "Saba":     Jins("Saba",     [0, 6.0, 13, 18],   ["T", "Eb½ (Bayati)", "F", "Gb"]),
}

# ──────────────────────────────────────────────
# Maqam definitions
# ──────────────────────────────────────────────

@dataclass
class MaqamTemplate:
    name: str
    tonic_abs: float          # tonic in comma-space from C4
    tonic_name: str
    lower_jins: str
    upper_base_abs: float
    upper_default: str
    upper_options: list[str]

MAQAM = {
    "Rast on C":     MaqamTemplate("Rast on C",     PC["C"],              "C",    "Rast",     PC["G"], "Rast",     ["Rast","Bayati","Hijaz","Nahawand"]),
    "Bayati on D":   MaqamTemplate("Bayati on D",   PC["D"],              "D",    "Bayati",   PC["G"], "Rast",     ["Rast","Nahawand","Hijaz"]),
    "Nahawand on C": MaqamTemplate("Nahawand on C", PC["C"],              "C",    "Nahawand", PC["G"], "Nahawand", ["Rast","Bayati","Hijaz","Nahawand"]),
    "Ajam on C":     MaqamTemplate("Ajam on C",     PC["C"],              "C",    "Ajam",     PC["G"], "Ajam",     ["Ajam"]),
    "Kurd on D":     MaqamTemplate("Kurd on D",     PC["D"],              "D",    "Kurd",     PC["G"], "Kurd",     ["Kurd"]),
    "Hijaz on D":    MaqamTemplate("Hijaz on D",    PC["D"],              "D",    "Hijaz",    PC["G"], "Hijaz",    ["Hijaz"]),
    "Saba on D":     MaqamTemplate("Saba on D",     PC["D"],              "D",    "Saba",     PC["F"], "Hijaz",    ["Hijaz"]),
    "Sikah on Eb½":  MaqamTemplate("Sikah on Eb½",  MICRO["Eb_half_sikah"], "Eb½","Sikah",   PC["G"], "Hijaz",    ["Hijaz"]),
}

# ──────────────────────────────────────────────
# Scale builder (port of buildScaleOneOctave)
# ──────────────────────────────────────────────

@dataclass
class ScaleNote:
    abs_commas: float
    label: str
    is_tonic: bool
    is_micro: bool
    freq_hz: float = field(init=False)

    def __post_init__(self):
        self.freq_hz = comma_to_freq(self.abs_commas)

def _label_from_abs(abs_commas: float) -> str:
    """Map an absolute comma value to a note label string."""
    rel = abs_commas % OCT
    tol = 1e-6

    def close(a, b): return abs(a - b) < tol

    # Natural notes
    for name, val in PC.items():
        if name == "C5": continue
        if close(rel, val % OCT): return name

    # Accidentals
    Eb = PC["D"] + 4
    Fsh = PC["F"] + CHROM
    Gb = PC["F"] + DIAT
    Ab = PC["G"] + DIAT
    Bb = PC["A"] + DIAT
    Db = PC["C"] + DIAT
    Csh = PC["D"] - DIAT

    if close(rel, Eb):  return "Eb"
    if close(rel, Fsh): return "F#"
    if close(rel, Gb):  return "Gb"
    if close(rel, Ab):  return "Ab"
    if close(rel, Bb):  return "Bb"
    if close(rel, Db):  return "Db"
    if close(rel, Csh): return "C#"

    # Microtones
    for name, val in MICRO.items():
        if close(rel, val % OCT):
            return name

    return f"({rel:.1f}c)"

def build_scale(maqam_name: str, upper_jins_override: Optional[str] = None) -> list[ScaleNote]:
    """
    Build the full one-octave scale for a given maqam.
    Returns a list of ScaleNote objects sorted ascending.
    """
    m = MAQAM[maqam_name]
    upper_j = upper_jins_override or m.upper_default

    # Special case: Saba
    if m.lower_jins == "Saba":
        saba_offsets = [0, 6.0, 13, 17, 31, 35, 44, OCT]
        notes = []
        for i, off in enumerate(saba_offsets):
            abs_c = m.tonic_abs + off
            notes.append(ScaleNote(
                abs_commas=abs_c,
                label=_label_from_abs(abs_c),
                is_tonic=(i == 0),
                is_micro="½" in _label_from_abs(abs_c),
            ))
        return notes

    lower = JINS[m.lower_jins]
    upper = JINS[upper_j]

    lower_abs = [m.tonic_abs + o for o in lower.offsets]
    upper_abs = [m.upper_base_abs + o for o in upper.offsets]

    joined = list(lower_abs)
    tol = 1e-9
    base = upper_abs[0]
    has_base = any(abs(x - base) < tol for x in joined)
    to_add = upper_abs[1:] if has_base else upper_abs
    for x in to_add:
        if not any(abs(y - x) < tol for y in joined):
            joined.append(x)

    joined.append(m.tonic_abs + OCT)

    # Filter to one octave
    filtered = [x for x in joined if m.tonic_abs - tol <= x <= m.tonic_abs + OCT + tol]
    filtered.sort()

    # Deduplicate
    uniq = []
    for x in filtered:
        if not any(abs(y - x) < tol for y in uniq):
            uniq.append(x)

    # Sikah + Hijaz leading tone
    if m.lower_jins == "Sikah" and upper_j == "Hijaz":
        d_abs = PC["D"] + OCT
        if m.tonic_abs - tol <= d_abs <= m.tonic_abs + OCT + tol:
            if not any(abs(x - d_abs) < tol for x in uniq):
                uniq.append(d_abs)
                uniq.sort()

    # Trim to 8 notes
    if len(uniq) <= 8:
        result = uniq[:8]
    else:
        octave_abs = m.tonic_abs + OCT
        inside = [x for x in uniq if abs(x - octave_abs) > tol]
        picked = inside[:6]
        top = inside[-1]
        if not any(abs(y - top) < tol for y in picked):
            picked.append(top)
        picked.append(octave_abs)
        picked.sort()
        dedup = []
        for x in picked:
            if not any(abs(y - x) < tol for y in dedup):
                dedup.append(x)
        result = dedup[:8]

    return [
        ScaleNote(
            abs_commas=abs_c,
            label=_label_from_abs(abs_c),
            is_tonic=(i == 0),
            is_micro="½" in _label_from_abs(abs_c),
        )
        for i, abs_c in enumerate(result)
    ]

# ──────────────────────────────────────────────
# Pitch quantizer
# ──────────────────────────────────────────────

def quantize_to_maqam(freq_hz: float, scale: list[ScaleNote],
                       octave_range: int = 3) -> Optional[ScaleNote]:
    """
    Given a detected frequency, find the nearest scale degree
    across multiple octaves. Returns the best-matching ScaleNote
    (with abs_commas adjusted for octave).
    """
    if freq_hz <= 0:
        return None

    input_commas = freq_to_commas(freq_hz)
    best = None
    best_dist = float("inf")

    for note in scale:
        for oct_shift in range(-octave_range, octave_range + 1):
            shifted = note.abs_commas + oct_shift * OCT
            dist = abs(input_commas - shifted)
            if dist < best_dist:
                best_dist = dist
                # Return a copy with adjusted commas
                best = ScaleNote(
                    abs_commas=shifted,
                    label=note.label,
                    is_tonic=note.is_tonic,
                    is_micro=note.is_micro,
                )

    return best

def get_quantization_error_cents(freq_hz: float, quantized: ScaleNote) -> float:
    """Return the quantization error in cents (100 cents = 1 semitone)."""
    target_freq = comma_to_freq(quantized.abs_commas)
    return 1200 * math.log2(freq_hz / target_freq)


# ──────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=== 53-EDO Tuning Engine Self-Test ===\n")

    print("Microtonal reference frequencies:")
    for name, commas in MICRO.items():
        print(f"  {name:25s} = {commas:.1f}c → {comma_to_freq(commas):.2f} Hz")

    print("\nRast on C scale:")
    scale = build_scale("Rast on C")
    for n in scale:
        print(f"  {n.label:20s} {n.abs_commas:5.1f}c  {n.freq_hz:.2f} Hz")

    print("\nBayati on D scale:")
    scale = build_scale("Bayati on D")
    for n in scale:
        print(f"  {n.label:20s} {n.abs_commas:5.1f}c  {n.freq_hz:.2f} Hz")

    print("\nIqa'at (Rhythmic Cycles):")
    for name, iqa in IQA_AT.items():
        if iqa["type"] == "free":
            print(f"  {name:15s} - {iqa['name']} ({iqa['type']})")
        else:
            pattern_str = " ".join(str(p) for p in iqa["pattern"])
            print(f"  {name:15s} - {iqa['name']} [{pattern_str}] {iqa['time_signature']}")

    print("\nQuantization test (318.22 Hz → should be Eb½ Rast):")
    scale = build_scale("Rast on C")
    result = quantize_to_maqam(318.22, scale)
    err = get_quantization_error_cents(318.22, result)
    print(f"  Detected: {result.label}, error: {err:.2f} cents")

    print("\nQuantization test (316.15 Hz → should be Eb½ Bayati):")
    scale = build_scale("Bayati on D")
    result = quantize_to_maqam(316.15, scale)
    err = get_quantization_error_cents(316.15, result)
    print(f"  Detected: {result.label}, error: {err:.2f} cents")


def is_microtonal_interval(cents: float, tolerance: float = 15.0) -> bool:
    """
    Determine if a pitch interval is microtonal (not in 12-TET)

    Args:
        cents: Interval in cents between two pitches
        tolerance: How close to 12-TET to still consider "standard"

    Returns:
        True if the interval is likely microtonal
    """
    # Standard 12-TET intervals in cents
    standard_intervals = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]

    # Check if close to any standard interval
    for std in standard_intervals:
        if abs(cents - std) < tolerance:
            return False

    # Also check halfway points (50, 150, 250...)
    halfway = [50, 150, 250, 350, 450, 550, 650, 750, 850, 950, 1050, 1150]
    for half in halfway:
        if abs(cents - half) < tolerance:
            return True

    return True  # If not standard, it's microtonal


def analyze_pitch_for_microtones(
        pitch_sequence: np.ndarray,
        times: np.ndarray,
        scale: list[ScaleNote]
) -> list[dict]:
    """
    Analyze a pitch sequence to find microtonal passages
    """
    results = []

    for i in range(1, len(pitch_sequence)):
        if pitch_sequence[i] > 0 and pitch_sequence[i - 1] > 0:
            interval_cents = 1200 * np.log2(pitch_sequence[i] / pitch_sequence[i - 1])

            if is_microtonal_interval(interval_cents):
                results.append({
                    'time': times[i],
                    'freq': pitch_sequence[i],
                    'prev_freq': pitch_sequence[i - 1],
                    'interval_cents': interval_cents,
                    'is_microtonal': True
                })

    return results


def calibrate_scale_to_performer(scale: list['ScaleNote'], peaks_cents: np.ndarray) -> list['ScaleNote']:
    """
    SOTA Dynamic Tuning: Shifts the theoretical 53-EDO scale frequencies
    to perfectly match the performer's actual intonation peaks (KDE).
    """
    # Create a copy so we don't permanently mutate the mathematical constants
    calibrated_scale = copy.deepcopy(scale)
    C4 = 261.63  # Your base reference

    for note in calibrated_scale:
        # 1. Find where this note mathematically sits in the 0-1200 cent octave
        theoretical_cents = (1200 * math.log2(note.freq_hz / C4)) % 1200

        # 2. Find the closest human performer peak within a +/- 35 cent window
        closest_peak = None
        min_dist = 35.0

        for peak in peaks_cents:
            # Calculate circular distance (handles wrap-around at C, e.g., 1195 to 5 cents)
            dist = min(abs(theoretical_cents - peak),
                       abs(theoretical_cents - peak + 1200),
                       abs(theoretical_cents - peak - 1200))

            if dist < min_dist:
                min_dist = dist
                closest_peak = peak

                # 3. Apply the human tuning offset
                if closest_peak is not None:
                    diff = closest_peak - theoretical_cents
                    if diff > 600: diff -= 1200
                    if diff < -600: diff += 1200

                    # Shift the exact frequency of this scale degree
                    note.freq_hz = note.freq_hz * (2 ** (diff / 1200.0))

                    # --- THE FIX: Update the comma math so the quantizer sees the shift! ---
                    note.abs_commas = freq_to_commas(note.freq_hz)

                    # We also update a debug label so we can see the offset
                    note.label = f"{note.label} ({diff:+.1f}c)"

    return calibrated_scale