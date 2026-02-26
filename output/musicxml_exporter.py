"""
MusicXML Exporter

Converts quantized notes into a MusicXML file with proper microtonal
accidentals (quarter-tone flats and sharps) that can be opened in
Sibelius, MuseScore, Finale, or other notation software.

MusicXML microtonal encoding:
- Quarter-tone flat (♭½):  <accidental>three-quarters-flat</accidental>
  Actually we use: <accidental>flat</accidental> with tuning-alter -50 cents
  for maximum software compatibility.

Standard Arabic notation uses:
  - ♭½ (half-flat / quarter-tone flat): alter = -0.5 in MusicXML
  - #½ (half-sharp): alter = 0.5

Dependencies: music21 (for MusicXML generation)
"""

from fractions import Fraction
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import math

from core.rhythm_quantizer import QuantizedNote


@dataclass
class MXLNote:
    step: str  # C D E F G A B
    alter: float  # 0 = natural, -1 = flat, 1 = sharp, -0.5 = half-flat
    octave: int
    duration_type: str  # "quarter", "eighth", etc.
    duration_dots: int  # number of dots
    accidental_display: Optional[str] = None


# Fixed for standard MusicXML 4.0 representation
ACCIDENTAL_TEXT = {
    -1.0: "flat",
    0.0: "natural",
    1.0: "sharp",
    -0.5: "quarter-flat",  # The Arabic half-flat (slashed flat)
    0.5: "quarter-sharp",  # The Arabic half-sharp (crossed sharp)
}

DURATION_TYPE = {
    Fraction(1, 1): ("whole", 0),
    Fraction(3, 4): ("half", 1),
    Fraction(1, 2): ("half", 0),
    Fraction(3, 8): ("quarter", 1),
    Fraction(1, 4): ("quarter", 0),
    Fraction(3, 16): ("eighth", 1),
    Fraction(1, 8): ("eighth", 0),
    Fraction(1, 12): ("16th", 0), # Fallback for triplets
    Fraction(1, 16): ("16th", 0),
    Fraction(1, 32): ("32nd", 0),
}

NOTE_LABEL_TO_MXL = {
    # Natural notes
    "C": ("C", 0.0), "D": ("D", 0.0), "E": ("E", 0.0), "F": ("F", 0.0),
    "G": ("G", 0.0), "A": ("A", 0.0), "B": ("B", 0.0),

    # Standard Flats and Sharps
    "Eb": ("E", -1.0), "Ab": ("A", -1.0), "Bb": ("B", -1.0),
    "Db": ("D", -1.0), "Gb": ("G", -1.0),
    "F#": ("F", 1.0), "C#": ("C", 1.0), "G#": ("G", 1.0),

    # Quarter-Tone Flats (alter = -0.5)
    "Eb½ (Rast)": ("E", -0.5),
    "Eb½ (Bayati)": ("E", -0.5),
    "Eb½ (Sikah)": ("E", -0.5),
    "Bb½ (Rast)": ("B", -0.5),
    "Bb½ (Bayati)": ("B", -0.5),
    "Bb½ (Sikah)": ("B", -0.5),

    # --- OUR NEW SAFETY NETS ---
    "E half-flat": ("E", -0.5),
    "B half-flat": ("B", -0.5),
    "E_half_flat": ("E", -0.5),
    "B_half_flat": ("B", -0.5),
    "A_half_flat": ("A", -0.5),

    # Quarter-Tone Sharps (alter = 0.5)
    "F_half_sharp": ("F", 0.5),
    "C_half_sharp": ("C", 0.5),
    "G_half_sharp": ("G", 0.5),
    "Hijaz_E_quarter": ("E", 0.5),

    # Specific maqam inflections
    "Saba_F_flat": ("F", -1.0),
}


def freq_to_octave(freq_hz: float) -> int:
    """
    Mathematically calculate the octave, but clamp it to realistic
    instrument ranges to prevent visual octave-leaping from CQT artifacts.
    """
    if freq_hz <= 0:
        return 4

    semitones_from_c0 = 12 * math.log2(freq_hz / 16.3516)
    octave = int(semitones_from_c0 // 12)

    # Hard-clamp the octave between 3 and 5.
    return max(3, min(5, octave))


def label_to_mxl_note(note: QuantizedNote) -> Optional[MXLNote]:
    """Convert a QuantizedNote to an MXLNote structure."""
    label = note.note_label

    if label not in NOTE_LABEL_TO_MXL:
        # Sort keys by length descending!
        # This forces it to check "E_half_flat" BEFORE it checks "E".
        sorted_keys = sorted(NOTE_LABEL_TO_MXL.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if label.startswith(key):
                label = key
                break
        else:
            return None

    step, alter = NOTE_LABEL_TO_MXL[label]
    octave = freq_to_octave(note.median_freq)
    dur_type, dots = DURATION_TYPE.get(note.duration_beats, ("quarter", 0))

    return MXLNote(
        step=step,
        alter=alter,
        octave=octave,
        duration_type=dur_type,
        duration_dots=dots,
        accidental_display=ACCIDENTAL_TEXT.get(alter),
    )

# ──────────────────────────────────────────────
# MusicXML builder
# ──────────────────────────────────────────────

DIVISIONS = 16  # MusicXML divisions per quarter note

DURATION_BEATS_TO_DIVISIONS = {
    Fraction(1, 1): DIVISIONS * 4,
    Fraction(3, 4): DIVISIONS * 3,
    Fraction(1, 2): DIVISIONS * 2,
    Fraction(3, 8): int(DIVISIONS * 1.5),
    Fraction(1, 4): DIVISIONS,
    Fraction(3, 16): int(DIVISIONS * 0.75),
    Fraction(1, 8): DIVISIONS // 2,
    Fraction(1, 12): int(DIVISIONS / 1.5),
    Fraction(1, 16): DIVISIONS // 4,
    Fraction(1, 32): DIVISIONS // 8,
}


def build_musicxml(
        measures: list[list[QuantizedNote]],
        title: str = "Arabic Transcription",
        composer: str = "",
        maqam_name: str = "",
        bpm: float = 80.0,
        time_signature: tuple[int, int] = (4, 4),
) -> str:
    """
    Build a complete MusicXML document from quantized measures.
    Now with Grace Note (Ornament) Support!
    """
    lines = []

    # Header
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<!DOCTYPE score-partwise PUBLIC')
    lines.append('  "-//Recordare//DTD MusicXML 4.0 Partwise//EN"')
    lines.append('  "http://www.musicxml.org/dtds/partwise.dtd">')
    lines.append('<score-partwise version="4.0">')

    # Work title
    lines.append('  <work>')
    lines.append(f'    <work-title>{_esc(title)}</work-title>')
    lines.append('  </work>')

    # Identification
    lines.append('  <identification>')
    if composer:
        lines.append('    <creator type="composer">' + _esc(composer) + '</creator>')
    if maqam_name:
        lines.append('    <rights>Maqam: ' + _esc(maqam_name) + '</rights>')
    lines.append('    <encoding>')
    lines.append('      <software>Arabic Transcriber (SOTA Engine)</software>')
    lines.append('    </encoding>')
    lines.append('  </identification>')

    # Part list
    lines.append('  <part-list>')
    lines.append('    <score-part id="P1">')
    lines.append('      <part-name>Melody</part-name>')
    lines.append('    </score-part>')
    lines.append('  </part-list>')

    # Part
    lines.append('  <part id="P1">')

    for measure_idx, measure_notes in enumerate(measures):
        lines.append(f'    <measure number="{measure_idx + 1}">')

        # Attributes (first measure only)
        if measure_idx == 0:
            lines.append('      <attributes>')
            lines.append(f'        <divisions>{DIVISIONS}</divisions>')
            lines.append('        <key><fifths>0</fifths></key>')
            lines.append('        <time>')
            lines.append(f'          <beats>{time_signature[0]}</beats>')
            lines.append(f'          <beat-type>{time_signature[1]}</beat-type>')
            lines.append('        </time>')
            lines.append('        <clef>')
            lines.append('          <sign>G</sign>')
            lines.append('          <line>2</line>')
            lines.append('        </clef>')
            lines.append('      </attributes>')

            # Tempo marking
            lines.append('      <direction placement="above">')
            lines.append('        <direction-type>')
            lines.append('          <metronome parentheses="no">')
            lines.append('            <beat-unit>quarter</beat-unit>')
            lines.append(f'            <per-minute>{int(bpm)}</per-minute>')
            lines.append('          </metronome>')
            lines.append('        </direction-type>')
            lines.append(f'        <sound tempo="{int(bpm)}"/>')
            lines.append('      </direction>')

            # Maqam annotation
            if maqam_name:
                lines.append('      <direction placement="above">')
                lines.append('        <direction-type>')
                lines.append(f'          <words font-style="italic" font-weight="bold" default-y="20">Maqam: {_esc(maqam_name)}</words>')
                lines.append('        </direction-type>')
                lines.append('      </direction>')

        # Process Notes (and their ornaments)
        for note in measure_notes:

            # 1. PROCESS GRACE NOTES (ORNAMENTS) FIRST
            for ornament in note.ornaments_before:
                mxl_ornament = label_to_mxl_note(ornament)
                if mxl_ornament is None:
                    continue

                lines.append('      <note>')
                lines.append('        <grace slash="yes"/>')  # This makes it a grace note!
                lines.append('        <pitch>')
                lines.append(f'          <step>{mxl_ornament.step}</step>')
                if mxl_ornament.alter != 0.0:
                    lines.append(f'          <alter>{mxl_ornament.alter}</alter>')
                lines.append(f'          <octave>{mxl_ornament.octave}</octave>')
                lines.append('        </pitch>')
                lines.append('        <type>16th</type>') # Grace notes are visually rendered as 16th notes
                if mxl_ornament.accidental_display:
                    lines.append(f'        <accidental>{mxl_ornament.accidental_display}</accidental>')
                lines.append('      </note>')

            # 2. PROCESS MAIN NOTE
            mxl = label_to_mxl_note(note)
            if mxl is None:
                continue

            div = DURATION_BEATS_TO_DIVISIONS.get(note.duration_beats, DIVISIONS)
            lines.append('      <note>')
            lines.append('        <pitch>')
            lines.append(f'          <step>{mxl.step}</step>')
            if mxl.alter != 0.0:
                lines.append(f'          <alter>{mxl.alter}</alter>')
            lines.append(f'          <octave>{mxl.octave}</octave>')
            lines.append('        </pitch>')
            lines.append(f'        <duration>{div}</duration>')
            lines.append(f'        <type>{mxl.duration_type}</type>')
            if mxl.duration_dots > 0:
                for _ in range(mxl.duration_dots):
                    lines.append('        <dot/>')
            if mxl.accidental_display:
                lines.append(f'        <accidental>{mxl.accidental_display}</accidental>')

            lines.append('      </note>')

        lines.append('    </measure>')

    lines.append('  </part>')
    lines.append('</score-partwise>')

    return "\n".join(lines)


def _esc(s: str) -> str:
    """XML escape."""
    return (s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def export_musicxml(
        measures: list[list[QuantizedNote]],
        output_path: str | Path,
        title: str = "Arabic Transcription",
        composer: str = "",
        maqam_name: str = "",
        bpm: float = 80.0,
        time_signature: tuple[int, int] = (4, 4),
) -> Path:
    """
    Write MusicXML to file.
    """
    xml = build_musicxml(
        measures, title, composer, maqam_name, bpm, time_signature
    )
    output_path = Path(output_path)
    output_path.write_text(xml, encoding="utf-8")
    print(f"      ✓ MusicXML saved: {output_path}")
    return output_path