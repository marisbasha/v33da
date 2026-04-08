"""Mel-spectrogram CNN encoder for the identity stream."""

import torch
import torch.nn as nn

from ...data.features import MelSpecExtractor


class MelSpecEncoder(nn.Module):
    """
    Simple CNN on mel-spectrogram for audio identity embedding.

    Input: (B, N_samples) single-channel waveform
    Output: (B, T, d_out) embedding sequence
    """

    def __init__(self, sample_rate=24414, n_mels=80, d_out=256):
        super().__init__()
        self.mel = MelSpecExtractor(sample_rate=sample_rate, n_mels=n_mels)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # After 3x stride-2 on mel axis: 80 -> 40 -> 20 -> 10
        self.proj = nn.Linear(128 * 10, d_out)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """(B, N_samples) -> (B, T, d_out)"""
        mel = self.mel(waveform)  # (B, n_mels, T)
        x = mel.unsqueeze(1)  # (B, 1, n_mels, T)
        x = self.cnn(x)  # (B, 128, n_mels//8, T)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)  # (B, T, 128*10)
        x = self.proj(x)  # (B, T, d_out)
        return x
