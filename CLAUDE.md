# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Monophonic Arabic music transcriber: audio file → MusicXML sheet music. The pipeline handles microtonal Arabic maqamat using 53-EDO (Equal Divisions of the Octave) tuning, which is incompatible with standard 12-TET. The output can be opened in MuseScore, Sibelius, or Finale.

## Running the Transcriber

```bash
# Activate venv first
source .venv/bin/activate

# Basic usage
python transcribe.py <audio_file> --output output.xml

# With options
python transcribe.py audio.wav --output out.xml --maqam "Bayati on D" --instrument oud

# Available instruments: voice_male, voice_female, oud, nay, violin, kanun, general
# Available maqamat: see core/tuning.py MAQAM dict keys

# Save pre-processed audio for debugging pitch issues
python transcribe.py audio.wav --save-cleaned-audio

# Suppress verbose pipeline output
python transcribe.py audio.wav --quiet

# Run tuning engine self-test
python core/tuning.py
```

## Installing Dependencies

```bash
pip install -r requirements.txt
```

PENN model weights (`fcnf0++.pt`) must be placed at the project root. They are excluded from git (large file). If the file is absent, PENN falls back to its bundled default checkpoint.

## Architecture

The pipeline runs as 7 sequential steps in `transcribe.py::transcribe()`:

1. **Audio load** (`core/audio_loader.py`) — librosa load → amplitude normalize → pydub fallback for exotic formats
2. **Pitch extraction** (`core/pitch_extractor.py`) — PENN neural tracker → interpolated medfilt (kernel=11) for octave-jump smoothing → `PitchTrack` dataclass
3. **Maqam detection** (`core/maqam_detector.py`) — KDE fingerprint of the pitch distribution → FFT circular cross-correlation against each theoretical maqam → returns ranked `MaqamCandidate` list with a `tuning_offset_cents` field
4. **Scale calibration** (`core/tuning.py::calibrate_scale_to_performer`) — shifts the theoretical 53-EDO scale grid to match the performer's actual intonation peaks (within ±35 cents per degree)
5. **Note segmentation** (`core/pitch_extractor.py::segment_notes_sota`) — "Islands and Anchors": voiced regions are sliced into notes when pitch drifts beyond `SEGMENTATION['drift_threshold_cents']` (45 cents by default)
6. **Rhythm quantization** (`core/rhythm_quantizer.py`) — floating-pulse taksim quantizer: median note duration → beat unit → snaps to `RHYTHMIC_VALUES` fractions
7. **MusicXML export** (`output/musicxml_exporter.py`) — hand-built MusicXML 4.0 string with quarter-tone accidentals (alter ±0.5)

## Key Domain Constraints (from AGENTS.md — enforce strictly)

- **Never quantize pitch to 12-TET MIDI notes.** All pitch is continuous Hz or cents; the only quantization target is the 53-EDO scale grid.
- **Never commit `.pt`, `.pth`, or `.h5` files** — model weights are gitignored.
- **Keep modules separated**: pitch extraction logic stays in `core/pitch_extractor.py`, tuning math in `core/tuning.py`, rhythm in `core/rhythm_quantizer.py`. Don't cross-contaminate.
- Prefer SOTA DSP approaches (median filtering, ensemble tracking, KDE) over naive moving averages for any pitch artifact fixes.

## Tuning System

All musical math is in **comma-space** (`core/tuning.py`):
- 1 octave = 53 commas (`OCT = 53`)
- `COMMA = 2^(1/53)` — one Holdrian comma
- `C4 = 261.63 Hz` — reference pitch
- `PC` dict: natural note positions in commas from C4
- `MICRO` dict: microtonal positions (e.g., `Eb_half_rast = 15.5c`)
- `MAQAM` dict: 8 defined maqamat — **Rast on C, Bayati on D, Nahawand on C, Ajam on C, Kurd on D, Hijaz on D, Saba on D, Sikah on Eb½**
- `build_scale(maqam_name)` assembles lower + upper jins into a one-octave `ScaleNote` list
- `quantize_to_maqam(freq_hz, scale)` finds nearest scale degree across ±3 octaves

## Configuration

All tunable parameters live in `core/config.py` — do not hardcode equivalents elsewhere:
- `PITCH_EXTRACTION` — hop size, fmin/fmax defaults, confidence threshold (0.35), device
- `SEGMENTATION` — min note duration (80ms), drift threshold (45 cents)
- `MAQAM_DETECTION` — KDE bandwidth (0.1), match window (45 cents), micro-note weight
- `RHYTHM_QUANTIZATION` — default BPM (90), taksim mode flag

## Test Audio

Sample files are in `tests/test_audio/`. There is no automated test suite; testing is done by running the full pipeline on these files and inspecting the MusicXML output in MuseScore.
