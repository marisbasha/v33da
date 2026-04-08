"""MUSIC baseline: Multiple Signal Classification → DOA → nearest bird."""

import numpy as np
from .utils import nearest_bird, SPEED_OF_SOUND_MM_S


class MUSICBaseline:
    """
    MUSIC algorithm for 3D sound source localization.

    Compute spatial covariance from multi-channel STFT, eigendecompose
    to find noise subspace, scan steering vectors over 3D grid.
    """

    def __init__(self, mic_positions: np.ndarray, sample_rate: float = 24414.0,
                 grid_resolution: int = 20, n_fft: int = 1024, n_sources: int = 1,
                 cage_bounds: tuple = ((-90.8, 1155.2), (-1563.9, -667.9), (236.0, 1530.0))):
        self.mic_positions = mic_positions
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_sources = n_sources
        self.n_mics = len(mic_positions)

        # Build search grid
        xs = np.linspace(cage_bounds[0][0], cage_bounds[0][1], grid_resolution)
        ys = np.linspace(cage_bounds[1][0], cage_bounds[1][1], grid_resolution)
        zs = np.linspace(cage_bounds[2][0], cage_bounds[2][1], grid_resolution)
        self.grid_flat = np.stack(
            np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1
        ).reshape(-1, 3)

    def _steering_vector(self, pos: np.ndarray, freq: float) -> np.ndarray:
        """Compute steering vector for a given position and frequency."""
        dists = np.linalg.norm(self.mic_positions - pos, axis=1)  # (n_mics,)
        delays = dists / SPEED_OF_SOUND_MM_S  # seconds
        return np.exp(-2j * np.pi * freq * delays)  # (n_mics,)

    def predict(self, waveforms: np.ndarray, bird_positions: np.ndarray) -> dict:
        """
        Args:
            waveforms: (n_mics, N_samples)
            bird_positions: (N_birds, 3) in mm
        """
        n_mics, n_samples = waveforms.shape

        # STFT
        hop = self.n_fft // 2
        n_frames = max(1, (n_samples - self.n_fft) // hop + 1)
        window = np.hanning(self.n_fft)

        stft_frames = []
        for t in range(n_frames):
            start = t * hop
            end = start + self.n_fft
            if end > n_samples:
                break
            frame = waveforms[:, start:end] * window  # (n_mics, n_fft)
            stft_frames.append(np.fft.rfft(frame, axis=1))  # (n_mics, F)

        if not stft_frames:
            # Fallback: random
            return {
                "predicted_bird_idx": 0,
                "predicted_position": self.mic_positions.mean(axis=0),
            }

        stft_frames = np.array(stft_frames)  # (T, n_mics, F)
        n_freqs = stft_frames.shape[2]

        # Spatial covariance matrix (averaged across time and frequency)
        R = np.zeros((n_mics, n_mics), dtype=complex)
        for t in range(len(stft_frames)):
            for f_idx in range(n_freqs):
                x = stft_frames[t, :, f_idx]  # (n_mics,)
                R += np.outer(x, x.conj())
        R /= len(stft_frames) * n_freqs

        # Eigendecompose
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        # Noise subspace: smallest eigenvalues
        noise_subspace = eigenvectors[:, :n_mics - self.n_sources]  # (n_mics, n_mics-n_sources)

        # MUSIC pseudo-spectrum: scan grid
        # Use a representative frequency (center of vocalization band ~4kHz)
        center_freq = 4000.0
        spectrum = np.zeros(len(self.grid_flat))

        for g in range(len(self.grid_flat)):
            a = self._steering_vector(self.grid_flat[g], center_freq)  # (n_mics,)
            a = a / np.linalg.norm(a)
            # MUSIC: 1 / (a^H @ En @ En^H @ a)
            proj = noise_subspace.conj().T @ a  # (n_noise,)
            denom = np.real(proj.conj() @ proj)
            spectrum[g] = 1.0 / (denom + 1e-10)

        peak_idx = np.argmax(spectrum)
        estimated_pos = self.grid_flat[peak_idx]
        bird_idx, dist = nearest_bird(estimated_pos, bird_positions)

        return {
            "predicted_bird_idx": bird_idx,
            "predicted_position": estimated_pos,
            "music_peak_value": float(spectrum[peak_idx]),
            "nearest_dist_mm": dist,
        }
