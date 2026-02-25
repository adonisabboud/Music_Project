"""
Configuration settings for Arabic Music Transcriber
All tunable parameters in one place
"""

# Pitch extraction
PITCH_EXTRACTION = {
    'cqt_confidence_threshold': 0.3,
    'cqt_energy_threshold_db': -40,
    'hop_length': 512,
    'bins_per_octave': 53,
}

# Rhythm quantization
RHYTHM_QUANTIZATION = {
    'iqa_detection_enabled': True,
}
