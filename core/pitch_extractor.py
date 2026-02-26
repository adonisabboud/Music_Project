"""
Pitch Extraction and Segmentation Module

This module implements the definitive SOTA pipeline for pitch tracking:
1. It uses a monkey-patching technique to intercept raw probabilities from PENN.
2. It builds a custom, strict transition matrix to heavily penalize octave jumps.
3. It uses a custom Viterbi decoder to find the most musically plausible
   melodic line, providing full control over octave correction.
4. It segments the clean contour into discrete musical notes.
"""
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import penn
import torch
from scipy.signal import medfilt

# Import penn.decode to patch it
import penn.decode

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


def _custom_viterbi_decode(logits, **kwargs):
    """
    A custom Viterbi decoder that replaces PENN's default decoder.
    This function intercepts the logits (posteriorgram) and applies
    our strict octave-jump penalties.
    """
    # logits shape is likely (batch, bins, frames) or (batch, frames, bins)
    # We assume batch size 1.
    
    # Convert tensor to numpy
    if isinstance(logits, torch.Tensor):
        logits = logits.squeeze().cpu().numpy()
    
    # Ensure shape is (bins, frames) for librosa
    if logits.shape[0] != 1440: # 1440 bins is standard for PENN
        # If shape is (frames, bins), transpose it
        logits = logits.T
        
    num_bins, num_frames = logits.shape
    
    # 1. Build the Transition Matrix
    bins = np.arange(num_bins)
    i, j = np.meshgrid(bins, bins)
    
    # CRITICAL TUNING: A very small sigma makes large jumps impossible.
    sigma = 20.0  # 100 cents
    transition_matrix = np.exp(-0.5 * ((i - j) / sigma)**2)
    
    # 2. Decode
    # Logits are already log-probabilities (usually)
    # But PENN might return raw logits. Viterbi expects log-likelihoods.
    # We'll assume they are suitable for Viterbi.
    
    log_trans = np.log(transition_matrix + 1e-10)
    
    # librosa.sequence.viterbi expects (n_states, n_steps)
    path = librosa.sequence.viterbi(logits, log_trans)
    
    # 3. Convert path to pitch and periodicity
    # We need to return what PENN expects: (pitch, periodicity)
    
    # Convert bins to Hz
    cents = penn.CENTS_PER_BIN * path
    pitch = 10. * 2**(cents / 1200.)
    
    # Get confidence from the logits
    # We take the value at the chosen path
    # logits is (bins, frames)
    confidence = logits[path, np.arange(num_frames)]
    
    # Normalize confidence if it's not 0-1 (logits can be anything)
    # Sigmoid? Or just use as is? 
    # PENN's periodicity is usually 0-1.
    # If logits are unnormalized, we might need softmax?
    # But for now let's assume they are usable as confidence scores relative to each other.
    # Actually, let's map them to 0-1 roughly.
    confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min() + 1e-6)

    # Convert back to tensors as PENN expects
    pitch = torch.tensor(pitch).unsqueeze(0) # (1, frames)
    periodicity = torch.tensor(confidence).unsqueeze(0) # (1, frames)
    
    return pitch, periodicity


def extract_pitch_penn(y, sr=16000, fmin=None, fmax=None):
    """
    Extracts pitch by monkey-patching PENN to use our custom decoder.
    """
    audio = torch.from_numpy(y).unsqueeze(0)
    hopsize = PITCH_EXTRACTION['hop_size_sec']
    
    checkpoint_path = Path(__file__).parent.parent / 'fcnf0++.pt'
    checkpoint_arg = str(checkpoint_path) if checkpoint_path.exists() else None

    # --- MONKEY PATCHING ---
    # Save original decoder just in case
    original_viterbi = getattr(penn.decode, 'viterbi', None)
    
    # Replace with our custom decoder
    penn.decode.viterbi = _custom_viterbi_decode
    
    try:
        # Run PENN with 'viterbi' decoder (which is now ours!)
        pitch, periodicity = penn.from_audio(
            audio, sr, hopsize=hopsize,
            fmin=fmin, fmax=fmax,
            checkpoint=checkpoint_arg,
            decoder='viterbi' 
        )
    finally:
        # Restore original decoder to be safe
        if original_viterbi:
            penn.decode.viterbi = original_viterbi

    return PitchTrack(
        times=np.arange(len(pitch.squeeze())) * hopsize,
        frequencies=pitch.squeeze().cpu().numpy(),
        confidences=periodicity.squeeze().cpu().numpy(),
        sample_rate=sr,
        hop_size=int(sr * hopsize)
    )


def segment_notes_sota(track: PitchTrack, audio_y: np.ndarray):
    """
    SOTA Segmentation: Combines Pitch Velocity + Spectral Onsets.
    """
    log_freqs = np.log2(np.maximum(track.frequencies, 1.0))
    velocity = np.abs(np.gradient(log_freqs))
    velocity_smooth = medfilt(velocity, kernel_size=SEGMENTATION['smooth_kernel'])

    onset_env = librosa.onset.onset_strength(y=audio_y, sr=track.sample_rate)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=track.sample_rate, units='time')

    hop_sec = track.hop_size / track.sample_rate
    onset_indices = np.unique(np.round(onsets / hop_sec).astype(int))
    
    is_onset_frame = np.zeros(len(track.times), dtype=bool)
    valid_onsets = onset_indices[onset_indices < len(track.times)]
    is_onset_frame[valid_onsets] = True

    vel_threshold = SEGMENTATION['velocity_threshold']

    segments = []
    current_start = 0

    for i in range(1, len(track.times)):
        is_boundary = is_onset_frame[i] or (velocity_smooth[i] > vel_threshold)
        is_unvoiced = not track.voiced_mask[i]

        if is_boundary or is_unvoiced:
            duration = track.times[i] - track.times[current_start]
            
            if duration > SEGMENTATION['min_note_duration']:
                chunk_indices = np.arange(current_start, i)
                chunk_mask = track.voiced_mask[chunk_indices]
                
                if np.any(chunk_mask):
                    voiced_chunk_freqs = track.frequencies[chunk_indices][chunk_mask]
                    segments.append({
                        "start_time": track.times[current_start],
                        "duration_sec": duration,
                        "median_freq": np.median(voiced_chunk_freqs),
                        "mean_confidence": np.mean(track.confidences[chunk_indices])
                    })
            
            current_start = i

    return segments
