"""
Arabic Music Transcriber — SOTA Pipeline with PENN + Onset Segmentation
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.audio_loader import load_audio, get_duration
from core.pitch_extractor import extract_pitch_penn, segment_notes_sota
from core.maqam_detector import detect_maqam_with_consistency
from core.tuning import (
    build_scale, quantize_to_maqam, get_quantization_error_cents, MAQAM,
    calibrate_scale_to_performer
)
from core.instruments import get_instrument_range, INSTRUMENTS
from output.musicxml_exporter import export_musicxml
from core.rhythm_quantizer import quantize_rhythm_taksim, notes_to_measures


def transcribe(
    audio_path: str,
    output_path: str,
    pipeline_mode: str = "full",
    rhythm_mode: str = "auto",
    maqam_override: str = None,
    bpm_override: float = None,
    iqa_override: str = None,
    instrument: str = "general",
    time_signature: tuple = (4, 4),
    verbose: bool = True,
) -> dict:
    """
    Main Entry Point: SOTA transcription pipeline.
    
    Orchestrates the following steps:
    1. Audio Loading & Preprocessing.
    2. Neural Pitch Extraction with integrated Viterbi Octave Correction.
    3. Tuning-Invariant Maqam Detection.
    4. Note Segmentation.
    5. Rhythm Quantization.
    6. MusicXML Export.
    """

    def log(msg):
        if verbose:
            print(msg)

    # ── Step 1: Load audio & Pre-process ──────────────────────────────
    log(f"\n{'='*60}")
    log(f"  Arabic Music Transcriber (SOTA Engine)")
    log(f"{'='*60}")
    log(f"\n[1/6] Loading and cleaning audio: {audio_path}")

    y, sr = load_audio(audio_path, isolate_melody=True)
    duration = get_duration(y, sr)
    log(f"      Duration: {duration:.1f}s, Sample rate: {sr}Hz")
    log(f"      Applied spectral denoising and HPSS.")

    # ── Step 2: Pitch extraction (Custom Viterbi Decoder) ───────────────
    fmin, fmax = get_instrument_range(instrument)
    log(f"\n[2/6] Extracting pitch with custom Viterbi decoder...")
    log(f"      Instrument: {instrument} (range: {fmin:.0f}-{fmax:.0f} Hz)")

    # This function now contains the integrated octave correction logic
    track = extract_pitch_penn(y, sr=sr, fmin=fmin, fmax=fmax)
    log(f"      Voiced frames: {track.voiced_mask.sum() / len(track.voiced_mask):.1%}")
    
    # ── Step 3: Maqam Detection (Tuning-Invariant) ──────────────────────
    log(f"\n[3/6] Detecting Maqam (Tuning-Invariant)...")

    if maqam_override:
        log(f"      Override provided: {maqam_override}")
        maqam_name = maqam_override
    else:
        candidates = detect_maqam_with_consistency(list(track.frequencies), list(track.confidences))
        best_cand = candidates[0] if candidates else None
        if best_cand:
            maqam_name = best_cand.name
            tuning_offset = best_cand.tuning_offset_cents
            log(f"      Detected: {maqam_name} (Tuning: {tuning_offset:+.1f} cents)")
        else:
            maqam_name = "Rast on C"
            log(f"      Detection failed, falling back to {maqam_name}")

    # ── Step 4: Note Segmentation (Velocity + Onsets) ───────────────────
    log(f"\n[4/6] Segmenting notes (Glissando-aware)...")
    
    scale = build_scale(maqam_name)
    raw_segments = segment_notes_sota(track, y)
    log(f"      Main notes detected: {len(raw_segments)}")

    # ── Step 5: Tuning & Rhythm Quantization ────────────────────────────
    log(f"\n[5/6] Quantizing pitch and rhythm...")

    quantized_pitches = [
        quantize_to_maqam(seg["median_freq"], scale)
        for seg in raw_segments
    ]

    error_cents_list = []
    for seg, qnote in zip(raw_segments, quantized_pitches):
        if qnote is not None:
            err = get_quantization_error_cents(seg["median_freq"], qnote)
            error_cents_list.append(abs(err))

    if error_cents_list:
        avg_error = sum(error_cents_list) / len(error_cents_list)
        log(f"      Avg tuning error: {avg_error:.1f} cents")

    quantized_notes, bpm, detected_iqa = quantize_rhythm_taksim(
        raw_segments, quantized_pitches
    )
    log(f"      Estimated BPM: {bpm:.0f}")
    log(f"      Notes: {len(quantized_notes)}")

    # ── Step 6: MusicXML Export ─────────────────────────────────────────
    log(f"\n[6/6] Exporting MusicXML...")
    
    measures = notes_to_measures(quantized_notes, time_signature)
    log(f"      Measures: {len(measures)}")

    title = Path(audio_path).stem.replace("_", " ").title()
    export_musicxml(
        measures,
        output_path=output_path,
        title=title,
        maqam_name=maqam_name,
        bpm=bpm,
        time_signature=time_signature,
    )

    log(f"\n{'=' * 60}")
    log(f"  Done! Output: {output_path}")
    log(f"{'=' * 60}\n")

    if verbose:
        open_in_musescore(output_path)

    return {
        "maqam": maqam_name,
        "bpm": bpm,
        "iqa": detected_iqa,
        "note_count": len(quantized_notes),
        "measure_count": len(measures),
        "output_path": output_path,
        "avg_quantization_error_cents": (
            sum(error_cents_list) / len(error_cents_list) if error_cents_list else 0.0
        ),
    }


def open_in_musescore(file_path: str):
    """Attempts to open the generated XML file in MuseScore across platforms."""
    if not os.path.exists(file_path):
        print(f"      ⚠ File not found: {file_path}")
        return

    system = platform.system()
    try:
        if system == "Linux":
            musescore_cmds = ['musescore', 'musescore3', 'mscore', 'mscore3', 'musescore4']
            for cmd in musescore_cmds:
                try:
                    subprocess.run([cmd, file_path], check=True, capture_output=True)
                    print(f"      ✓ Opened in MuseScore ({cmd})")
                    return
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            print("      ⚠ MuseScore not found. Please install it or open manually.")
        elif system == "Windows":
            possible_paths = [
                r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
                r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe",
                r"C:\Program Files (x86)\MuseScore 3\bin\MuseScore3.exe",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    subprocess.run([path, file_path])
                    print(f"      ✓ Opened in MuseScore")
                    return
            os.startfile(file_path)
            print(f"      ✓ Opened with default XML viewer")
        elif system == "Darwin":
            subprocess.run(['open', '-a', 'MuseScore', file_path])
            print(f"      ✓ Opened in MuseScore")
    except Exception as e:
        print(f"      ⚠ Could not open MuseScore automatically: {e}")
        print(f"      File saved at: {file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Arabic music audio → MusicXML transcriber",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("audio", help="Input audio file")
    parser.add_argument("--output", "-o", default="output.xml", help="Output MusicXML file")

    parser.add_argument("--pipeline", "-p", default="full", choices=["mvp", "full"],
                        help="Pipeline mode")

    parser.add_argument("--mode", "-M", default="auto", choices=["auto", "metered", "taksim"],
                        help="Rhythm mode")
    parser.add_argument("--maqam", "-m", default=None, help="Override maqam")
    parser.add_argument("--iqa", default=None, help="Override iqa'")
    parser.add_argument("--instrument", "-i", default="general", choices=list(INSTRUMENTS.keys()), help="Instrument type")
    parser.add_argument("--bpm", "-b", type=float, default=None, help="Override tempo")
    parser.add_argument("--time-sig", "-t", default="4/4", help="Time signature")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")

    args = parser.parse_args()

    try:
        num, denom = args.time_sig.split("/")
        time_sig = (int(num), int(denom))
    except Exception:
        print(f"Invalid time signature: {args.time_sig}")
        sys.exit(1)

    result = transcribe(
        audio_path=args.audio,
        output_path=args.output,
        pipeline_mode=args.pipeline,
        rhythm_mode=args.mode,
        maqam_override=args.maqam,
        bpm_override=args.bpm,
        iqa_override=args.iqa,
        instrument=args.instrument,
        time_signature=time_sig,
        verbose=not args.quiet,
    )
    print(f"\nSummary:")
    print(f"  Maqam:      {result['maqam']}")
    print(f"  Iqa':       {result['iqa'] or 'None (taksim)'}")
    print(f"  BPM:        {result['bpm']:.0f}")
    print(f"  Notes:      {result['note_count']}")
    print(f"  Avg error:  {result['avg_quantization_error_cents']:.1f} cents")
    print(f"  Output:     {result['output_path']}")

if __name__ == "__main__":
    main()