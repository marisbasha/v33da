"""SRP-PHAT baseline: steered response power → nearest bird."""

import numpy as np
from .utils import nearest_bird, SPEED_OF_SOUND_MM_S


class SRPPHATBaseline:
    """
    Steered Response Power with PHAT weighting.

    Scan a 3D grid over the cage volume, compute steered response power
    at each point, pick the peak as the estimated source position.
    """

    def __init__(self, mic_positions: np.ndarray, sample_rate: float = 24414.0,
                 grid_resolution: int = 20, n_fft: int = 1024,
                 cage_bounds: tuple = ((-90.8, 1155.2), (-1563.9, -667.9), (236.0, 1530.0))):
        self.mic_positions = mic_positions
        self.sample_rate = sample_rate
        self.n_fft = n_fft

        # Build 3D search grid
        xs = np.linspace(cage_bounds[0][0], cage_bounds[0][1], grid_resolution)
        ys = np.linspace(cage_bounds[1][0], cage_bounds[1][1], grid_resolution)
        zs = np.linspace(cage_bounds[2][0], cage_bounds[2][1], grid_resolution)
        self.grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)  # (Gx, Gy, Gz, 3)
        self.grid_flat = self.grid.reshape(-1, 3)  # (G, 3)

        # Precompute delays from each grid point to each mic
        n_mics = len(mic_positions)
        # (G, n_mics) distances in mm
        dists = np.linalg.norm(
            self.grid_flat[:, None, :] - mic_positions[None, :, :], axis=2
        )
        # Convert to delays in samples
        self.delays = dists / SPEED_OF_SOUND_MM_S * sample_rate  # (G, n_mics)

    def predict(self, waveforms: np.ndarray, bird_positions: np.ndarray) -> dict:
        """
        Args:
            waveforms: (n_mics, N_samples)
            bird_positions: (N_birds, 3) in mm

        Returns:
            dict with predicted_bird_idx, predicted_position
        """
        n_mics, n_samples = waveforms.shape

        # Compute STFT for all mics
        window = np.hanning(self.n_fft)
        stfts = []
        for c in range(n_mics):
            # Zero-pad
            padded = np.zeros(self.n_fft)
            L = min(n_samples, self.n_fft)
            padded[:L] = waveforms[c, :L] * window[:L]
            stfts.append(np.fft.rfft(padded))
        stfts = np.array(stfts)  # (n_mics, F)

        # PHAT normalization
        stfts_phat = stfts / (np.abs(stfts) + 1e-10)

        # Compute SRP at each grid point
        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)  # (F,)
        n_freqs = len(freqs)

        # For each grid point, compute steered power
        srp = np.zeros(len(self.grid_flat))
        for g in range(len(self.grid_flat)):
            delays_g = self.delays[g]  # (n_mics,) in samples
            # Phase shift for steering: exp(-j * 2pi * f * delay)
            phase_shifts = np.exp(-2j * np.pi * freqs[None, :] * delays_g[:, None] / self.sample_rate)
            # Steered signal
            steered = stfts_phat * phase_shifts  # (n_mics, F)
            # Sum across mics and compute power
            srp[g] = np.abs(steered.sum(axis=0)).sum()

        # Peak
        peak_idx = np.argmax(srp)
        estimated_pos = self.grid_flat[peak_idx]

        # Nearest bird
        bird_idx, dist = nearest_bird(estimated_pos, bird_positions)

        return {
            "predicted_bird_idx": bird_idx,
            "predicted_position": estimated_pos,
            "srp_peak_value": float(srp[peak_idx]),
            "nearest_dist_mm": dist,
        }
