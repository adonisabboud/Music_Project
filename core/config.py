"""
Configuration settings for Arabic Music Transcriber
SOTA PENN + Onset-Gated Segmentation Parameters
"""

# ─── Pitch Extraction (PENN) ──────────────────────────────────────────
PITCH_EXTRACTION = {
    'hop_size_sec': 0.01,         # 10ms frames for high temporal resolution
    'fmin': 50.0,                 # Min freq (low Oud G is ~78Hz)
    'fmax': 1600.0,               # Max freq (high Nay/Violin register)
    'decoder': 'viterbi',         # 'viterbi' for musicality, 'argmax' for speed
    'confidence_threshold': 0.1,  # PENN periodicity threshold (0.0 to 1.0)
    'device': 'auto',             # 'cuda', 'cpu', or 'auto'
}

# ─── Note Segmentation (SOTA) ─────────────────────────────────────────
# These settings control how the continuous pitch "ribbon" is chopped
# into discrete notes for the MusicXML score.
SEGMENTATION = {
    'velocity_threshold': 0.02,   # Sensitivity to pitch slides (lower = more notes)
    'min_note_duration': 0.08,    # Minimum note length in seconds (~32nd note)
    'smooth_kernel': 5,           # Median filter size to ignore vibrato jitter
    'onset_weight': 1.0,          # How much to trust physical string plucks
}

# ─── Maqam & Tuning ───────────────────────────────────────────────────
MAQAM_DETECTION = {
    'match_window_cents': 35.0,   # How close a note must be to a scale degree
    'micro_weight': 2.5,          # Importance of microtonal degrees in scoring
    'kde_bandwidth': 0.1,         # Smoothing for the Dynamic Tuning histogram
}

# ─── Rhythm Quantization ──────────────────────────────────────────────
RHYTHM_QUANTIZATION = {
    'iqa_detection_enabled': True,
    'default_bpm': 90.0,
    'max_tempo_drift': 0.15,      # Allow 15% BPM variation for rubato/taksim
}