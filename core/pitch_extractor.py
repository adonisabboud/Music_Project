from dataclasses import dataclass

import librosa
import numpy as np
import penn
import torch
from scipy.signal import medfilt


@dataclass
class PitchTrack:
    times: np.ndarray
    frequencies: np.ndarray
    confidences: np.ndarray
    sample_rate: int
    hop_size: int

    @property
    def voiced_mask(self) -> np.ndarray:
        return self.confidences > 0.1  # PENN confidence threshold


def extract_pitch_penn(y, sr=16000):
    """SOTA Pitch Extraction using PENN with Viterbi decoding."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    audio = torch.from_numpy(y).unsqueeze(0)

    # 10ms frames are the SOTA standard for temporal precision
    hopsize = 0.01

    # We use 'viterbi' to ensure the contour is a connected musical thought
    pitch, periodicity = penn.from_audio(
        audio, sr, hopsize=hopsize,
        fmin=50.0, fmax=1600.0,
        decoder='viterbi', device=device
    )

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
    Optimized for glissandos and microtonal transitions.
    """
    # 1. Calculate Pitch Velocity (Derivative of log-frequency)
    # Log-space is essential because we perceive pitch logarithmically
    log_freqs = np.log2(np.maximum(track.frequencies, 1.0))
    velocity = np.abs(np.gradient(log_freqs))

    # Smooth velocity to ignore tiny vibrato jitters
    velocity_smooth = medfilt(velocity, kernel_size=5)

    # 2. Get Spectral Onsets (Physical Plucks)
    onset_env = librosa.onset.onset_strength(y=audio_y, sr=track.sample_rate)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=track.sample_rate, units='time')

    # 3. Combine Boundaries
    # A boundary exists if velocity spikes OR an onset is detected
    vel_threshold = 0.02  # Adjusted for expressive slides
    potential_boundaries = track.times[velocity_smooth > vel_threshold]

    # Final segmentation logic
    segments = []
    current_start = 0

    # Simple logic to group frames into notes based on boundaries
    # (In a full implementation, we'd use HMM/Viterbi here too)
    for i in range(1, len(track.times)):
        is_boundary = (track.times[i] in onsets) or (velocity_smooth[i] > vel_threshold)
        is_unvoiced = track.confidences[i] < 0.1

        if is_boundary or is_unvoiced:
            duration = track.times[i] - track.times[current_start]
            if duration > 0.08:  # Min note duration (approx a 32nd note at 120bpm)
                chunk = track.frequencies[current_start:i]
                voiced_chunk = chunk[track.confidences[current_start:i] > 0.1]

                if len(voiced_chunk) > 0:
                    segments.append({
                        "start_time": track.times[current_start],
                        "duration_sec": duration,
                        "median_freq": np.median(voiced_chunk),
                        "mean_confidence": np.mean(track.confidences[current_start:i])
                    })
            current_start = i

    return segments