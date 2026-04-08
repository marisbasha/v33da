"""Accelerometer spectrogram oracle baselines for ceiling analysis."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AccelOracleNet(nn.Module):
    """Shared 2D encoder over per-bird accelerometer spectrograms."""

    def __init__(self, hidden_dim: int = 96, n_fft: int = 256, hop_length: int = 64):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        accel = batch["accelerometer"]  # (B, N, T)
        bird_mask = batch["bird_mask"]
        accel_mask = batch["accel_mask"]
        bird_positions = batch["bird_positions"]

        bsz, max_birds, n_samples = accel.shape
        accel = accel * accel_mask[:, None, :].float()
        accel_centered = accel - accel.mean(dim=-1, keepdim=True)
        accel_scale = accel_centered.std(dim=-1, keepdim=True).clamp_min(1e-4)
        accel_norm = (accel_centered / accel_scale).clamp(-8.0, 8.0)

        x = accel_norm.reshape(bsz * max_birds, n_samples)
        window = torch.hann_window(self.n_fft, device=x.device)
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )
        spec = spec.abs().pow(2)
        spec = torch.log1p(spec)

        spec_mean = spec.mean(dim=(-2, -1), keepdim=True)
        spec_std = spec.std(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
        spec = (spec - spec_mean) / spec_std
        feat = self.encoder(spec.unsqueeze(1))

        mean_feat = feat.mean(dim=(-2, -1))
        max_feat = feat.amax(dim=(-2, -1))
        bird_feat = torch.cat([mean_feat, max_feat], dim=-1).reshape(bsz, max_birds, -1)

        logits = self.score_head(bird_feat).squeeze(-1)
        logits = logits.masked_fill(~bird_mask, -1e9)
        pred_idx = logits.argmax(dim=-1)
        pred_position = bird_positions[torch.arange(bsz, device=bird_positions.device), pred_idx]
        return {"logits": logits, "pred_position": pred_position}


def accel_oracle_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return nn.functional.cross_entropy(out["logits"], batch["label_idx"])
