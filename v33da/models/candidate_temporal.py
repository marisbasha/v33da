"""Candidate branch with temporal radio processing via 1D CNN."""

import torch
import torch.nn as nn


class RadioTemporalEncoder(nn.Module):
    """
    Encode raw radio temporal signals per bird using 1D CNN.

    Input: (B, N_birds, n_channels, n_frames) raw radio signals
    Output: (B, N_birds, d_out) per-bird radio embedding
    """

    def __init__(self, n_channels: int = 7, d_out: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(64, d_out)

    def forward(self, radio_temporal: torch.Tensor) -> torch.Tensor:
        """(B, N_birds, n_ch, n_frames) -> (B, N_birds, d_out)"""
        B, N, C, T = radio_temporal.shape
        # Process all birds as batch
        x = radio_temporal.reshape(B * N, C, T)
        x = self.cnn(x)  # (B*N, 64, 1)
        x = x.squeeze(-1)  # (B*N, 64)
        x = self.proj(x)  # (B*N, d_out)
        return x.reshape(B, N, -1)


class JointRadioTemporalEncoder(nn.Module):
    """
    Encode radio jointly across all birds and all frames.

    Input: (B, N_birds, n_channels, n_frames)
    Output:
        bird_emb: (B, N_birds, d_out) per-bird embedding informed by all birds
        global_emb: (B, d_out) global radio context for the whole sample
    """

    def __init__(self, n_channels: int = 7, d_out: int = 64, hidden: int = 64, n_heads: int = 4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, hidden, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
        )
        self.bird_pos = nn.Parameter(torch.randn(1, 8, hidden) * 0.02)
        self.bird_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=n_heads,
                dim_feedforward=hidden * 2,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=1,
        )
        self.bird_proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_out),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, radio_temporal: torch.Tensor, bird_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            radio_temporal: (B, N_birds, n_channels, n_frames)
            bird_mask: (B, N_birds) bool
        """
        B, N, C, T = radio_temporal.shape
        if bird_mask is None:
            bird_mask = torch.ones(B, N, dtype=torch.bool, device=radio_temporal.device)

        masked_radio = radio_temporal * bird_mask[:, :, None, None].to(radio_temporal.dtype)
        x = masked_radio.permute(0, 2, 1, 3)  # (B, C, N, T)
        x = self.cnn(x)  # (B, H, N, T)

        per_bird = x.mean(dim=-1).transpose(1, 2)  # (B, N, H)
        if self.bird_pos.shape[1] < N:
            extra = N - self.bird_pos.shape[1]
            pos_pad = torch.zeros(1, extra, self.bird_pos.shape[2], device=per_bird.device, dtype=per_bird.dtype)
            bird_pos = torch.cat([self.bird_pos.to(per_bird.dtype).to(per_bird.device), pos_pad], dim=1)
        else:
            bird_pos = self.bird_pos[:, :N].to(per_bird.dtype).to(per_bird.device)
        per_bird = per_bird + bird_pos

        per_bird = self.bird_attn(per_bird, src_key_padding_mask=~bird_mask)

        weights = bird_mask.to(per_bird.dtype).unsqueeze(-1)
        global_ctx = (per_bird * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        global_expand = global_ctx.unsqueeze(1).expand(-1, N, -1)
        bird_emb = self.bird_proj(torch.cat([per_bird, global_expand], dim=-1))
        bird_emb = bird_emb * weights
        global_emb = self.global_proj(global_ctx)
        return bird_emb, global_emb


class CandidateEncoderTemporal(nn.Module):
    """
    Candidate encoder with raw temporal radio signals.

    Position + head orient + temporal radio CNN embedding → query.
    """

    def __init__(self, modality: str = "audio_3d_radio", d_model: int = 256,
                 dropout: float = 0.1, n_radio_channels: int = 7):
        super().__init__()
        self.modality = modality

        # Position + head features
        if modality == "audio_only":
            d_static = 3
        elif modality == "audio_3d":
            d_static = 6
        elif modality == "audio_3d_radio":
            d_static = 6  # position + head, radio handled separately
        else:
            raise ValueError(f"Unknown modality: {modality}")

        self.static_norm = nn.BatchNorm1d(d_static)

        if modality == "audio_3d_radio":
            self.radio_encoder = RadioTemporalEncoder(
                n_channels=n_radio_channels, d_out=d_model // 2,
            )
            d_combined = d_static + d_model // 2
        else:
            self.radio_encoder = None
            d_combined = d_static

        self.mlp = nn.Sequential(
            nn.Linear(d_combined, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, bird_positions: torch.Tensor, bird_head_orient: torch.Tensor,
                bird_radio: torch.Tensor, bird_radio_temporal: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            bird_positions: (B, N_birds, 3)
            bird_head_orient: (B, N_birds, 3)
            bird_radio: (B, N_birds, 12) summary stats (unused if temporal available)
            bird_radio_temporal: (B, N_birds, n_ch, n_frames) raw signals

        Returns:
            (B, N_birds, d_model)
        """
        B, N, _ = bird_positions.shape

        if self.modality == "audio_only":
            static = bird_positions
        else:
            static = torch.cat([bird_positions, bird_head_orient], dim=-1)

        # Normalize static features
        D = static.shape[-1]
        static = self.static_norm(static.reshape(B * N, D)).reshape(B, N, D)

        if self.radio_encoder is not None and bird_radio_temporal is not None:
            radio_emb = self.radio_encoder(bird_radio_temporal)  # (B, N, d_model//2)
            features = torch.cat([static, radio_emb], dim=-1)
        else:
            features = static

        return self.mlp(features)
