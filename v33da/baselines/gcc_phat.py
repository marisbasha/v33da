"""GCC-PHAT baseline: TDOA estimation → triangulation → nearest bird."""

import numpy as np
from ..data.features import gcc_phat
from .utils import nearest_bird, triangulate_tdoa, get_mic_pairs, SPEED_OF_SOUND_MM_S


class GCCPHATBaseline:
    """
    Generalized Cross-Correlation with Phase Transform.

    For each mic pair, estimate TDOA via GCC-PHAT peak.
    Triangulate to 3D position, assign to nearest bird.
    """

    def __init__(self, mic_positions: np.ndarray, sample_rate: float = 24414.0):
        """
        Args:
            mic_positions: (n_mics, 3) mic 3D coordinates in mm
            sample_rate: audio sample rate in Hz
        """
        self.mic_positions = mic_positions
        self.sample_rate = sample_rate
        self.mic_pairs = get_mic_pairs(len(mic_positions))

        # Max delay in samples based on max mic distance
        max_dist = 0
        for i, j in self.mic_pairs:
            d = np.linalg.norm(mic_positions[i] - mic_positions[j])
            max_dist = max(max_dist, d)
        self.max_delay = int(max_dist / SPEED_OF_SOUND_MM_S * sample_rate) + 10

    def predict(self, waveforms: np.ndarray, bird_positions: np.ndarray) -> dict:
        """
        Args:
            waveforms: (n_mics, N_samples)
            bird_positions: (N_birds, 3) in mm

        Returns:
            dict with predicted_bird_idx, predicted_position, tdoas
        """
        # Estimate TDOA for each mic pair
        tdoas = []
        for i, j in self.mic_pairs:
            _, tdoa_samples = gcc_phat(waveforms[i], waveforms[j], max_delay=self.max_delay)
            tdoa_sec = tdoa_samples / self.sample_rate
            tdoas.append(tdoa_sec)

        tdoas = np.array(tdoas)

        # Triangulate
        estimated_pos = triangulate_tdoa(tdoas, self.mic_pairs, self.mic_positions)

        # Nearest bird
        bird_idx, dist = nearest_bird(estimated_pos, bird_positions)

        return {
            "predicted_bird_idx": bird_idx,
            "predicted_position": estimated_pos,
            "tdoas": tdoas,
            "nearest_dist_mm": dist,
        }
