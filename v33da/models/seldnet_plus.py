"""SELDnet+ — SELDnet with TDOA conditioning, RFN, and radio fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..data.features import MelSpecExtractor
from ..data.dataset import MIC_POSITIONS, CAGE_X, CAGE_Y, CAGE_Z

SPEED_OF_SOUND = 343000.0


class RelaxedFreqNorm(nn.Module):
    """Relaxed frequency-wise normalization for cross-domain robustness."""
    def __init__(self, n_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, n_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, n_channels, 1, 1))
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        mu = x.mean(dim=2, keepdim=True)
        sigma = x.std(dim=2, keepdim=True) + 1e-6
        x_norm = (x - mu) / sigma
        a = torch.sigmoid(self.alpha)
        return (a * x + (1 - a) * x_norm) * self.gamma + self.beta


class SELDnetPlus(nn.Module):
    """
    SELDnet+ = SELDnet audio encoder + TDOA-conditioned per-bird scoring + radio.

    Key differences from vanilla SELDnet:
    - RFN normalization for cross-experiment robustness
    - Per-bird scoring conditioned on TDOA (physics-grounded)
    - Optional radio temporal features
    - Augmentation-friendly (mic dropout handled externally)
    """

    def __init__(
        self,
        n_mics: int = 5,
        sample_rate: int = 24414,
        n_mels: int = 80,
        hidden_dim: int = 128,
        use_radio: bool = True,
        n_radio_channels: int = 7,
    ):
        super().__init__()
        self.n_mics = n_mics
        self.use_radio = use_radio
        self.mel = MelSpecExtractor(sample_rate=sample_rate, n_mels=n_mels)

        self.register_buffer('mic_pos', torch.tensor(MIC_POSITIONS, dtype=torch.float32))
        bounds = torch.tensor([CAGE_X, CAGE_Y, CAGE_Z], dtype=torch.float32)
        self.register_buffer('cage_lo', bounds[:, 0])
        self.register_buffer('cage_hi', bounds[:, 1])

        # CNN with RFN
        self.rfn_input = RelaxedFreqNorm(n_mics)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_mics, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((4, 1)),
            RelaxedFreqNorm(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((4, 1)),
            RelaxedFreqNorm(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((5, 1)),
        )

        # BiGRU
        self.gru = nn.GRU(
            256, hidden_dim, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.1,
        )
        audio_dim = hidden_dim * 2  # 256

        # TDOA spatial encoder
        tdoa_dim = n_mics - 1  # 4
        self.tdoa_encoder = nn.Sequential(
            nn.Linear(tdoa_dim, 64),
            nn.ReLU(),
            nn.Linear(64, audio_dim),
        )

        # Radio temporal encoder (optional)
        if use_radio:
            self.radio_cnn = nn.Sequential(
                nn.Conv1d(n_radio_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.radio_proj = nn.Linear(64, audio_dim)
            scorer_dim = audio_dim * 3  # audio + tdoa + radio
        else:
            scorer_dim = audio_dim * 2  # audio + tdoa

        # Per-bird scorer
        self.scorer = nn.Sequential(
            nn.Linear(scorer_dim, audio_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(audio_dim, audio_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(audio_dim // 2, 1),
        )

        # Localization from audio (auxiliary)
        self.loc_head = nn.Sequential(
            nn.Linear(audio_dim, audio_dim // 2),
            nn.ReLU(),
            nn.Linear(audio_dim // 2, 3),
            nn.Sigmoid(),
        )

    def _compute_tdoa(self, bird_pos_norm):
        pos_mm = bird_pos_norm * (self.cage_hi - self.cage_lo) + self.cage_lo
        dists = torch.cdist(pos_mm, self.mic_pos.unsqueeze(0).expand(pos_mm.shape[0], -1, -1))
        tdoa = (dists[:, :, 1:] - dists[:, :, :1]) / SPEED_OF_SOUND
        return tdoa  # (B, N, 4)

    def forward(self, batch: dict) -> dict:
        audio = batch["audio"]  # (B, n_mics, N)
        B = audio.shape[0]

        # Multi-channel mel spectrograms
        mels = []
        for mi in range(self.n_mics):
            mel = self.mel(audio[:, mi])  # (B, n_mels, T)
            mels.append(mel)
        x = torch.stack(mels, dim=1)  # (B, n_mics, n_mels, T)

        # RFN + CNN
        x = self.rfn_input(x)
        x = self.cnn(x)  # (B, 256, 1, T)
        x = x.squeeze(2).permute(0, 2, 1)  # (B, T, 256)

        # BiGRU
        x, _ = self.gru(x)  # (B, T, 256)
        audio_emb = x.mean(dim=1)  # (B, 256)

        # Localization (auxiliary, from audio only)
        pred_pos = self.loc_head(audio_emb)  # (B, 3)

        # TDOA spatial prior per bird
        tdoa = self._compute_tdoa(batch["bird_positions"])  # (B, N, 4)
        N_birds = tdoa.shape[1]
        tdoa_emb = self.tdoa_encoder(tdoa)  # (B, N, 256)

        # Radio temporal per bird (optional)
        if self.use_radio and "bird_radio_temporal" in batch:
            rt = batch["bird_radio_temporal"]  # (B, N, C, T)
            rt_flat = rt.reshape(B * N_birds, rt.shape[2], rt.shape[3])
            radio_feat = self.radio_cnn(rt_flat).squeeze(-1)  # (B*N, 64)
            radio_emb = self.radio_proj(radio_feat).reshape(B, N_birds, -1)
        else:
            radio_emb = None

        # Per-bird scoring: [audio_global; tdoa_per_bird; radio_per_bird]
        audio_expanded = audio_emb.unsqueeze(1).expand(-1, N_birds, -1)
        if radio_emb is not None:
            scorer_input = torch.cat([audio_expanded, tdoa_emb, radio_emb], dim=-1)
        else:
            scorer_input = torch.cat([audio_expanded, tdoa_emb], dim=-1)

        logits = self.scorer(scorer_input).squeeze(-1)  # (B, N)

        # Mask
        if "bird_mask" in batch:
            mask = batch["bird_mask"]
            n_batch = mask.shape[1]
            if n_batch < N_birds:
                logits = logits[:, :n_batch]
            elif n_batch > N_birds:
                pad = torch.full((B, n_batch - N_birds), float("-inf"), device=logits.device)
                logits = torch.cat([logits, pad], dim=1)
                n_batch_actual = mask.shape[1]
            logits = logits.masked_fill(~mask, float("-inf"))

        attn = F.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "pred_position": pred_pos,
            "attn_weights": attn,
        }
