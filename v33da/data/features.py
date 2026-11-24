"""Audio feature extraction: mel-spectrogram, SALSA-Lite, GCC-PHAT."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MelSpecExtractor(nn.Module):
    """Extract mel-spectrogram from single-channel audio."""

    def __init__(self, sample_rate=24414, n_mels=80, n_fft=1024, hop_length=256, f_min=500, f_max=12000):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max

        # Use torchaudio if available, else manual
        try:
            import torchaudio
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max,
            )
        except ImportError:
            self.mel_transform = None

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, N_samples) single-channel audio

        Returns:
            (B, n_mels, T) log-mel-spectrogram
        """
        if self.mel_transform is not None:
            mel = self.mel_transform(waveform)  # (B, n_mels, T)
        else:
            # Manual STFT + mel filterbank
            spec = torch.stft(
                waveform, self.n_fft, self.hop_length,
                return_complex=True, window=torch.hann_window(self.n_fft, device=waveform.device),
            )
            mel = spec.abs().pow(2)  # power spectrogram
            # Simple linear mel scale (approximate)
            mel = mel[:, :self.n_mels, :]

        # Log scale
        mel = torch.clamp(mel, min=1e-10).log()
        return mel


class SALSALiteExtractor(nn.Module):
    """
    SALSA-Lite: Spatial cue-augmented log-spectrogram.

    For each non-reference mic, compute normalized inter-channel phase difference (IPD)
    w.r.t. reference mic (mic 0). Stack with log-mel of reference channel.

    Output: (B, 1 + n_mics-1, T, F) = (B, n_mics, T, F) where first channel is log-mel.
    """

    def __init__(self, n_mics=7, sample_rate=24414, n_fft=1024, hop_length=256,
                 n_mels=80, f_min=500, f_max=12000, ref_mic=0):
        super().__init__()
        self.n_mics = n_mics
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.ref_mic = ref_mic

        self.mel_extractor = MelSpecExtractor(
            sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft,
            hop_length=hop_length, f_min=f_min, f_max=f_max,
        )

    def forward(self, multi_ch_audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            multi_ch_audio: (B, n_mics, N_samples)

        Returns:
            (B, n_mics, T, F) — channel 0 = log-mel, channels 1..6 = IPD cos/sin
        """
        B, C, N = multi_ch_audio.shape
        device = multi_ch_audio.device

        # STFT for all channels
        window = torch.hann_window(self.n_fft, device=device)
        stfts = []
        for c in range(C):
            s = torch.stft(
                multi_ch_audio[:, c], self.n_fft, self.hop_length,
                return_complex=True, window=window,
            )  # (B, F, T)
            stfts.append(s)

        # Reference channel log-mel
        ref_mel = self.mel_extractor(multi_ch_audio[:, self.ref_mic])  # (B, n_mels, T)

        # IPD for each non-reference mic
        ref_stft = stfts[self.ref_mic]  # (B, F, T)
        ipds = []
        for c in range(C):
            if c == self.ref_mic:
                continue
            # Normalized IPD: angle(X_c * conj(X_ref))
            phase_diff = stfts[c] * ref_stft.conj()
            phase_diff = phase_diff / (phase_diff.abs() + 1e-8)
            # Take cos of angle (real part of normalized cross-spectrum)
            ipd = phase_diff.real  # (B, F, T)
            # Truncate to n_mels frequency bins for alignment
            ipd = ipd[:, :self.n_mels, :]
            ipds.append(ipd)

        # Stack: (B, 1+n_non_ref, n_mels, T) -> transpose to (B, C, T, F)
        features = torch.stack([ref_mel] + ipds, dim=1)  # (B, n_mics, n_mels, T)
        features = features.permute(0, 1, 3, 2)  # (B, n_mics, T, n_mels)

        return features


def gcc_phat(sig1: np.ndarray, sig2: np.ndarray, max_delay: int | None = None) -> tuple[np.ndarray, int]:
    """
    Generalized Cross-Correlation with Phase Transform.

    Args:
        sig1, sig2: (N,) arrays
        max_delay: max lag to search (samples)

    Returns:
        (gcc, tdoa): cross-correlation array and estimated TDOA in samples
    """
    n = len(sig1) + len(sig2) - 1
    n_fft = 1 << (n - 1).bit_length()  # next power of 2

    S1 = np.fft.rfft(sig1, n=n_fft)
    S2 = np.fft.rfft(sig2, n=n_fft)

    cross = S1 * np.conj(S2)
    cross /= np.abs(cross) + 1e-10  # PHAT weighting

    gcc = np.fft.irfft(cross, n=n_fft)
    gcc = np.concatenate([gcc[-(n_fft // 2):], gcc[:n_fft // 2 + 1]])

    center = n_fft // 2
    if max_delay is not None:
        gcc_search = gcc[center - max_delay:center + max_delay + 1]
        tdoa = np.argmax(gcc_search) - max_delay
    else:
        tdoa = np.argmax(gcc) - center

    return gcc, tdoa
