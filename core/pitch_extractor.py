"""
Pitch Extraction and Segmentation Module

This module implements the definitive SOTA pipeline for pitch tracking:
1. It uses a monkey-patching technique to intercept raw probabilities from PENN.
2. It applies a "Temporal Prior" (Helper Vector) to boost continuity.
3. It builds a custom, strict transition matrix to heavily penalize octave jumps.
4. It uses a custom Viterbi decoder to find the most musically plausible melodic line.
5. It runs pYIN as a secondary "consensus" tracker.
6. It enforces octave agreement between PENN and pYIN.
7. It applies a final "sticky hysteresis" pass, anchored by the global median pitch.
8. It segments the clean contour into discrete musical notes.
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


def apply_temporal_prior(posteriorgram, boost_strength=0.3, decay_factor=0.8, sigma=15):
    """
    Applies a running temporal prior (helper vector) to a neural network's 
    pitch probability matrix to prevent octave jumps.
    """
    # Ensure posteriorgram is (frames, bins)
    if posteriorgram.shape[0] == 1440:
        posteriorgram = posteriorgram.T
        
    num_frames, num_bins = posteriorgram.shape
    modified_posteriorgram = np.copy(posteriorgram)
    
    # Initialize the blank helper vector
    helper_vector = np.zeros(num_bins)
    
    # Pre-calculate a standard bin index array
    bin_indices = np.arange(num_bins)
    
    for i in range(num_frames):
        # 1. Add the current helper "memory" to the raw neural network output
        modified_posteriorgram[i] += helper_vector
        
        # 2. Find the new winning pitch bin after the boost is applied
        winning_bin = np.argmax(modified_posteriorgram[i])
        
        # 3. Create a new Gaussian bell curve centered exactly on the winning bin
        new_gaussian = boost_strength * np.exp(-((bin_indices - winning_bin)**2) / (2 * sigma**2))
        
        # 4. Update the helper vector for the NEXT frame:
        helper_vector = (helper_vector * decay_factor) + new_gaussian
        
    return modified_posteriorgram


def _custom_viterbi_decode(logits, **kwargs):
    """
    A custom Viterbi decoder that replaces PENN's default decoder.
    """
    # Convert tensor to numpy
    if isinstance(logits, torch.Tensor):
        logits = logits.squeeze().cpu().numpy()
    
    # Ensure shape is (bins, frames) for librosa
    if logits.shape[0] != 1440: 
        logits = logits.T
        
    # --- STEP 1: Apply Temporal Prior (Helper Vector) ---
    # We transpose to (frames, bins) for the helper function, then back
    logits_T = logits.T
    logits_boosted_T = apply_temporal_prior(logits_T)
    logits = logits_boosted_T.T
        
    num_bins, num_frames = logits.shape
    
    # --- STEP 2: Viterbi Decoding ---
    bins = np.arange(num_bins)
    i, j = np.meshgrid(bins, bins)
    
    sigma = 20.0  # 100 cents
    transition_matrix = np.exp(-0.5 * ((i - j) / sigma)**2)
    
    log_trans = np.log(transition_matrix + 1e-10)
    
    # librosa.sequence.viterbi expects (n_states, n_steps)
    path = librosa.sequence.viterbi(logits, log_trans)
    
    # --- STEP 3: Convert to Output ---
    cents = penn.CENTS_PER_BIN * path
    pitch = 10. * 2**(cents / 1200.)
    
    confidence = logits[path, np.arange(num_frames)]
    confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min() + 1e-6)

    pitch = torch.tensor(pitch).unsqueeze(0) 
    periodicity = torch.tensor(confidence).unsqueeze(0) 
    
    return pitch, periodicity


def extract_pitch_pyin(y, sr, fmin, fmax, hop_length):
    """
    Runs pYIN (Probabilistic YIN) to get a robust, albeit less precise, pitch track.
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        hop_length=hop_length,
        fill_na=0.0
    )
    f0 = np.nan_to_num(f0)
    return f0


def apply_sticky_hysteresis(pitch_hz, global_median_hz, cent_tolerance=150):
    """
    Applies frame-by-frame sticky hysteresis, anchored by the global median pitch.
    """
    cleaned_pitch = np.copy(pitch_hz)
    
    # Initialize the anchor with the global median of the entire piece.
    # This provides a strong, stable starting point.
    trusted_anchor = global_median_hz if global_median_hz > 0 else 0.0
    
    for i in range(len(cleaned_pitch)):
        current_hz = cleaned_pitch[i]
        
        # If unvoiced, reset the anchor.
        if current_hz <= 0.0:
            trusted_anchor = global_median_hz if global_median_hz > 0 else 0.0
            continue
            
        # If anchor is not set (e.g., song starts with silence), set it now.
        if trusted_anchor == 0.0:
            trusted_anchor = current_hz
            continue
            
        # Calculate the derivative (Distance in Cents)
        cent_diff = 1200.0 * np.log2(current_hz / trusted_anchor)
        
        # Catch and correct octave jumps
        if (1200 - cent_tolerance) <= cent_diff <= (1200 + cent_tolerance):
            cleaned_pitch[i] = current_hz / 2.0
            trusted_anchor = cleaned_pitch[i]
        elif (-1200 - cent_tolerance) <= cent_diff <= (-1200 + cent_tolerance):
            cleaned_pitch[i] = current_hz * 2.0
            trusted_anchor = cleaned_pitch[i]
        else:
            # Normal melodic movement: update the anchor
            trusted_anchor = current_hz
            
    return cleaned_pitch


def extract_pitch_penn(y, sr=16000, fmin=None, fmax=None):
    """
    Extracts pitch using Ensemble Tracking (PENN + pYIN) + Sticky Hysteresis.
    """
    audio = torch.from_numpy(y).unsqueeze(0)
    hopsize = PITCH_EXTRACTION['hop_size_sec']
    hop_length = int(hopsize * sr)
    
    fmin = fmin or PITCH_EXTRACTION['fmin']
    fmax = fmax or PITCH_EXTRACTION['fmax']
    
    checkpoint_path = Path(__file__).parent.parent / 'fcnf0++.pt'
    checkpoint_arg = str(checkpoint_path) if checkpoint_path.exists() else None

    # --- 1. Run PENN (with custom Viterbi + Temporal Prior) ---
    original_viterbi = getattr(penn.decode, 'viterbi', None)
    penn.decode.viterbi = _custom_viterbi_decode
    
    try:
        pitch, periodicity = penn.from_audio(
            audio, sr, hopsize=hopsize,
            fmin=fmin, fmax=fmax,
            checkpoint=checkpoint_arg,
            decoder='viterbi' 
        )
    finally:
        if original_viterbi:
            penn.decode.viterbi = original_viterbi

    penn_freqs = pitch.squeeze().cpu().numpy()
    penn_confs = periodicity.squeeze().cpu().numpy()
    times = np.arange(len(penn_freqs)) * hopsize

    # --- 2. Run pYIN (The Gatekeeper) ---
    pyin_freqs = extract_pitch_pyin(y, sr, fmin, fmax, hop_length)
    
    min_len = min(len(penn_freqs), len(pyin_freqs))
    penn_freqs, penn_confs, pyin_freqs, times = [arr[:min_len] for arr in [penn_freqs, penn_confs, pyin_freqs, times]]

    # --- 3. Apply Consensus Logic ---
    consensus_freqs = penn_freqs.copy()
    for i in range(min_len):
        p_freq, y_freq = penn_freqs[i], pyin_freqs[i]
        if p_freq > 0 and y_freq > 0:
            ratio = p_freq / y_freq
            if 1.8 < ratio < 2.2: consensus_freqs[i] /= 2.0
            elif 3.6 < ratio < 4.4: consensus_freqs[i] /= 4.0
            elif 0.45 < ratio < 0.55: consensus_freqs[i] *= 2.0

    # --- 4. Apply Sticky Hysteresis (Anchored by Global Median) ---
    voiced_consensus = consensus_freqs[consensus_freqs > 0]
    global_median_hz = np.median(voiced_consensus) if len(voiced_consensus) > 0 else 0.0
    
    final_freqs = apply_sticky_hysteresis(consensus_freqs, global_median_hz)

    return PitchTrack(
        times=times,
        frequencies=final_freqs,
        confidences=penn_confs,
        sample_rate=sr,
        hop_size=hop_length
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
