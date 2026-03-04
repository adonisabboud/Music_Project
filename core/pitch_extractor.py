"""
Pitch Extraction and Segmentation Module

This module implements a robust SOTA pipeline for pitch tracking:
1. It gets the raw pitch contour from PENN.
2. It applies a median filter to eliminate sudden octave jumps.
3. It uses a bulletproof "Islands and Anchors" algorithm to segment notes
   based on pitch stability, which is ideal for legato taksim.
"""
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import penn
import torch
from scipy.signal import medfilt

from .config import PITCH_EXTRACTION, SEGMENTATION


@dataclass
class PitchTrack:
    """
    Container for pitch tracking results.
    """
    times: np.ndarray
    frequencies: np.ndarray
    confidences: np.ndarray
    sample_rate: int
    hop_size: int

    @property
    def voiced_mask(self) -> np.ndarray:
        return self.confidences > PITCH_EXTRACTION['confidence_threshold']


def extract_pitch_penn(y, sr=16000, fmin=None, fmax=None):
    """
    Extracts pitch using PENN and applies a median filter for octave correction.
    """
    audio = torch.from_numpy(y).unsqueeze(0)
    hopsize = PITCH_EXTRACTION['hop_size_sec']
    hop_length = int(hopsize * sr)
    
    fmin = fmin or PITCH_EXTRACTION['fmin']
    fmax = fmax or PITCH_EXTRACTION['fmax']
    
    checkpoint_path = Path(__file__).parent.parent / 'fcnf0++.pt'
    checkpoint_arg = str(checkpoint_path) if checkpoint_path.exists() else None

    # 1. Get the raw pitch from PENN
    raw_pitch_hz, periodicity = penn.from_audio(
        audio, sr, hopsize=hopsize,
        fmin=fmin, fmax=fmax,
        checkpoint=checkpoint_arg,
        decoder='argmax' 
    )
    
    raw_pitch_hz = raw_pitch_hz.squeeze().cpu().numpy()
    periodicity = periodicity.squeeze().cpu().numpy()

    # 2. Crush onset spikes with a median filter.
    # Fill unvoiced (zero) frames with interpolated values first so they don't
    # pull voiced frames near silence toward zero, then restore zeros after.
    voiced_mask = raw_pitch_hz > 0
    if voiced_mask.any():
        idx = np.arange(len(raw_pitch_hz))
        fill = np.interp(idx, idx[voiced_mask], raw_pitch_hz[voiced_mask])
    else:
        fill = raw_pitch_hz.copy()
    cleaned_pitch_hz = medfilt(fill, kernel_size=11)

    # 3. Zero out the unvoiced frames
    cleaned_pitch_hz[periodicity < PITCH_EXTRACTION['confidence_threshold']] = 0.0

    return PitchTrack(
        times=np.arange(len(cleaned_pitch_hz)) * hopsize,
        frequencies=cleaned_pitch_hz,
        confidences=periodicity,
        sample_rate=sr,
        hop_size=hop_length
    )


def segment_notes_sota(track: PitchTrack):
    """
    Bulletproof 'Islands and Anchors' Segmentation.
    Finds contiguous voiced phrases, then slices them internally based on cent deviation.
    """
    segments = []
    hop_sec = track.hop_size / track.sample_rate
    min_frames = int(SEGMENTATION['min_note_duration'] / hop_sec)
    
    # 1. Find the edges of the voiced "Islands"
    padded_voiced = np.pad(track.voiced_mask, (1, 1), mode='constant', constant_values=False)
    edges = np.diff(padded_voiced.astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    
    # 2. Process each Island
    for start, end in zip(starts, ends):
        island_len = end - start
        if island_len < min_frames:
            continue  # Island is too short, skip it
            
        island_freqs = track.frequencies[start:end]
        island_times = track.times[start:end]
        island_confs = track.confidences[start:end]
        
        current_note_start = 0
        anchor_freq = island_freqs[0]
        
        # 3. Walk through the island and slice when pitch drifts
        for i in range(1, island_len):
            # Skip unvoiced frames within the island
            if island_freqs[i] <= 0:
                continue
            
            # Calculate distance from the anchor in cents
            cents_diff = 1200.0 * np.log2(island_freqs[i] / anchor_freq)
            
            # Did the pitch drift to a new note?
            if abs(cents_diff) > SEGMENTATION['drift_threshold_cents']:
                note_len = i - current_note_start
                
                # THE FIX: Only slice IF the chunk is long enough to be a real note!
                if note_len >= min_frames:
                    note_freqs = island_freqs[current_note_start:i]
                    valid_freqs = note_freqs[note_freqs > 0]
                    
                    if len(valid_freqs) > 0:
                        segments.append({
                            "start_time": island_times[current_note_start],
                            "duration_sec": note_len * hop_sec,
                            "median_freq": np.median(valid_freqs),
                            "mean_confidence": np.mean(island_confs[current_note_start:i])
                        })
                    
                    # Drop the new anchor ONLY after a successful slice
                    current_note_start = i
                    anchor_freq = island_freqs[i]
                    
                # If it drifted but the note is too short, DO NOTHING. 
                # The anchor stays the same, and the frames keep accumulating!

        # 4. Save the final note of the island
        final_note_len = island_len - current_note_start
        if final_note_len >= min_frames:
            note_freqs = island_freqs[current_note_start:]
            valid_freqs = note_freqs[note_freqs > 0]
            if len(valid_freqs) > 0:
                segments.append({
                    "start_time": island_times[current_note_start],
                    "duration_sec": final_note_len * hop_sec,
                    "median_freq": np.median(valid_freqs),
                    "mean_confidence": np.mean(island_confs[current_note_start:])
                })
            
    return segments
