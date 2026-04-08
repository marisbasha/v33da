"""SELDnet + Radio fusion — proves radio telemetry adds value."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.features import MelSpecExtractor


class SELDnetRadio(nn.Module):
    """
    SELDnet with per-bird radio features fused before the task heads.

    Audio branch: multi-channel mel → CNN → biGRU → audio embedding
    Radio branch: per-bird radio features (12D) → MLP → radio embedding
    Fusion: audio embedding + per-bird radio query → cross-attention → heads
    """

    def __init__(
        self,
        n_mics: int = 5,
        sample_rate: int = 24414,
        n_mels: int = 80,
        max_birds: int = 4,
        hidden_dim: int = 128,
        n_radio_features: int = 12,
    ):
        super().__init__()
        self.n_mics = n_mics
        self.max_birds = max_birds
        self.mel = MelSpecExtractor(sample_rate=sample_rate, n_mels=n_mels)

        # Audio CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(n_mics, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((4, 1)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((4, 1)),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((5, 1)),
        )

        self.gru = nn.GRU(
            256, hidden_dim, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.1,
        )

        # Radio per-bird encoder
        self.radio_encoder = nn.Sequential(
            nn.BatchNorm1d(n_radio_features),
            nn.Linear(n_radio_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

        # Cross-attention: radio queries attend to audio context
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2, num_heads=4, batch_first=True, dropout=0.1,
        )

        # Attribution head
        self.sed_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

        # Localization head (from audio only)
        self.doa_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

    def forward(self, batch: dict) -> dict:
        audio = batch["audio"]  # (B, n_mics, N)
        B = audio.shape[0]

        # Audio branch
        mels = []
        for mi in range(self.n_mics):
            mel = self.mel(audio[:, mi])
            mels.append(mel)
        x = torch.stack(mels, dim=1)  # (B, n_mics, n_mels, T)

        x = self.cnn(x)  # (B, 256, 1, T)
        x = x.squeeze(2).permute(0, 2, 1)  # (B, T, 256)

        audio_ctx, _ = self.gru(x)  # (B, T, hidden*2)

        # Localization from audio pooled
        audio_pooled = audio_ctx.mean(dim=1)  # (B, hidden*2)
        position = self.doa_head(audio_pooled)  # (B, 3)

        # Radio branch — per-bird features
        radio = batch["bird_radio"]  # (B, N_birds, 12)
        N_birds = radio.shape[1]

        # Encode radio per bird
        radio_flat = radio.reshape(B * N_birds, -1)  # (B*N, 12)
        radio_emb = self.radio_encoder(radio_flat)  # (B*N, hidden*2)
        radio_queries = radio_emb.reshape(B, N_birds, -1)  # (B, N_birds, hidden*2)

        # Cross-attention: each bird's radio query attends to audio context
        fused, _ = self.cross_attn(radio_queries, audio_ctx, audio_ctx)  # (B, N_birds, hidden*2)

        # Per-bird logits
        logits = self.sed_head(fused).squeeze(-1)  # (B, N_birds)

        # Mask
        if "bird_mask" in batch:
            mask = batch["bird_mask"]
            n_batch_birds = mask.shape[1]
            if n_batch_birds < self.max_birds:
                logits = logits[:, :n_batch_birds]
            elif n_batch_birds > self.max_birds:
                pad = torch.full((B, n_batch_birds - self.max_birds), float("-inf"), device=logits.device)
                logits = torch.cat([logits, pad], dim=1)
            logits = logits.masked_fill(~mask, float("-inf"))

        return {
            "logits": logits,
            "pred_position": position,
            "attn_weights": F.softmax(logits, dim=-1),
        }
