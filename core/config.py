"""
Configuration settings for Arabic Music Transcriber
SOTA PENN + Onset-Gated Segmentation Parameters
"""

# ─── Pitch Extraction (PENN) ──────────────────────────────────────────
PITCH_EXTRACTION = {
    'hop_size_sec': 0.01,         # 10ms frames for high temporal resolution
    'fmin': 50.0,                 # Min freq (low Oud G is ~78Hz)
    'fmax': 1600.0,               # Max freq (high Nay/Violin register)
    'decoder': 'argmax',          # Use 'argmax' for speed, as post-processing handles smoothing
    'confidence_threshold': 0.10, # The minimum confidence to consider a frame "voiced"
    'device': 'auto',             # 'cuda', 'cpu', or 'auto'
}

# ─── Note Segmentation (Islands and Anchors) ──────────────────────────
# These settings control how the continuous pitch "ribbon" is chopped
# into discrete notes using pitch stability and silence.
SEGMENTATION = {
    'min_note_duration': 0.08,      # Minimum note length in seconds (~32nd note)
    'drift_threshold_cents': 45.0,  # Slice a new note if pitch drifts by this amount (e.g., quarter-tone)
}

# ─── Maqam & Tuning ───────────────────────────────────────────────────
MAQAM_DETECTION = {
    'match_window_cents': 45.0,   # Widened to 45c to forgive expressive intonation
    'micro_weight': 3.0,          # Microtonal notes are highly diagnostic
    'kde_bandwidth': 0.1,         # Smoothing for the Dynamic Tuning histogram
}

# ─── Rhythm Quantization ──────────────────────────────────────────────
RHYTHM_QUANTIZATION = {
    'iqa_detection_enabled': False,
    'default_bpm': 90.0,
    'max_tempo_drift': 0.15,      # Allow 15% BPM variation for rubato/taksim
}