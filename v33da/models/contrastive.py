"""Contrastive model — learn audio-spatial embeddings that transfer across experiments.

The idea: learn a shared embedding space where
  embed(audio) ≈ embed(spatial_signature_of_vocalizer)

At test time: embed the audio, embed each candidate's spatial signature,
pick the candidate with highest similarity. No bird identity needed.

The spatial signature uses TDOA + mic energy pattern — physics-based features
that generalize across experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import combinations

from ..data.features import MelSpecExtractor
from ..data.dataset import MIC_POSITIONS, CAGE_X, CAGE_Y, CAGE_Z

SPEED_OF_SOUND = 343000.0


class AudioEncoder(nn.Module):
    """Multi-channel audio -> embedding."""

    def __init__(self, n_mics=5, sample_rate=24414, n_mels=80, d_embed=128):
        super().__init__()
        self.n_mics = n_mics
        self.mel = MelSpecExtractor(sample_rate=sample_rate, n_mels=n_mels)

        # Multi-channel mel -> CNN -> GRU -> embedding
        self.cnn = nn.Sequential(
            nn.Conv2d(n_mics, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((4, 1)),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((4, 1)),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((5, 1)),
        )
        self.gru = nn.GRU(256, d_embed, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=0.1)
        self.proj = nn.Linear(d_embed * 2, d_embed)

    def forward(self, audio):
        """(B, n_mics, N) -> (B, d_embed)"""
        mels = [self.mel(audio[:, mi]) for mi in range(self.n_mics)]
        x = torch.stack(mels, dim=1)
        x = self.cnn(x).squeeze(2).permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x.mean(dim=1)
        return F.normalize(self.proj(x), dim=-1)


class SpatialEncoder(nn.Module):
    """Bird position -> spatial signature embedding using TDOA + energy pattern."""

    def __init__(self, n_mics=5, d_embed=128):
        super().__init__()
        self.n_mics = n_mics
        self.mic_pairs = list(combinations(range(n_mics), 2))
        n_pairs = len(self.mic_pairs)

        self.register_buffer('mic_pos', torch.tensor(MIC_POSITIONS, dtype=torch.float32))
        bounds = torch.tensor([CAGE_X, CAGE_Y, CAGE_Z], dtype=torch.float32)
        self.register_buffer('cage_lo', bounds[:, 0])
        self.register_buffer('cage_hi', bounds[:, 1])

        # Input: TDOA (n_pairs) + energy pattern (n_mics) + raw position (3)
        input_dim = n_pairs + n_mics + 3
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, d_embed),
        )

    def forward(self, bird_pos_norm):
        """(B, N_birds, 3) -> (B, N_birds, d_embed)"""
        pos_mm = bird_pos_norm * (self.cage_hi - self.cage_lo) + self.cage_lo
        B, N, _ = pos_mm.shape

        # TDOA features
        tdoas = []
        for mi, mj in self.mic_pairs:
            d_i = torch.linalg.norm(pos_mm - self.mic_pos[mi], dim=-1)
            d_j = torch.linalg.norm(pos_mm - self.mic_pos[mj], dim=-1)
            tdoa = (d_i - d_j) / SPEED_OF_SOUND * 24414  # in samples
            tdoas.append(tdoa / 60.0)  # normalize
        tdoa_feat = torch.stack(tdoas, dim=-1)  # (B, N, n_pairs)

        # Energy pattern (1/r^2 normalized)
        dists = torch.cdist(pos_mm, self.mic_pos.unsqueeze(0).expand(B, -1, -1))
        energy = 1.0 / (dists ** 2 + 1e-6)
        energy = energy / (energy.sum(dim=-1, keepdim=True) + 1e-10)  # (B, N, n_mics)

        # Normalized position
        pos_norm_feat = bird_pos_norm  # (B, N, 3) already [0,1]

        features = torch.cat([tdoa_feat, energy, pos_norm_feat], dim=-1)
        emb = self.mlp(features)
        return F.normalize(emb, dim=-1)


class ContrastiveAttributor(nn.Module):
    """
    Contrastive learning for vocal attribution.

    Training: audio embedding should be close to the spatial embedding of the
    vocalizer's position and far from non-vocalizer positions.

    Inference: for each candidate, compute similarity between audio embedding
    and spatial embedding. Highest similarity = predicted vocalizer.

    Loss: InfoNCE / supervised contrastive.
    """

    def __init__(self, n_mics=5, sample_rate=24414, n_mels=80, d_embed=128,
                 temperature=0.1, use_radio=True, n_radio_channels=7):
        super().__init__()
        self.temperature = temperature
        self.use_radio = use_radio

        self.audio_encoder = AudioEncoder(n_mics, sample_rate, n_mels, d_embed)
        self.spatial_encoder = SpatialEncoder(n_mics, d_embed)

        # Radio adds to spatial embedding
        if use_radio:
            self.radio_encoder = nn.Sequential(
                nn.Conv1d(n_radio_channels, 32, 3, padding=1),
                nn.BatchNorm1d(32), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                nn.Linear(32, d_embed),
            )
            self.spatial_fuse = nn.Linear(d_embed * 2, d_embed)
        else:
            self.radio_encoder = None

        # Localization head
        self.loc_head = nn.Sequential(
            nn.Linear(d_embed, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

    def forward(self, batch):
        audio = batch["audio"]  # (B, n_mics, N)
        B = audio.shape[0]

        # Audio embedding
        audio_emb = self.audio_encoder(audio)  # (B, d_embed), L2-normalized

        # Spatial embeddings for each candidate bird
        spatial_emb = self.spatial_encoder(batch["bird_positions"])  # (B, N, d_embed)

        # Add radio to spatial embedding
        N_birds = spatial_emb.shape[1]
        if self.use_radio and "bird_radio_temporal" in batch:
            rt = batch["bird_radio_temporal"]
            rt_flat = rt.reshape(B * N_birds, rt.shape[2], rt.shape[3])
            radio_emb = self.radio_encoder(rt_flat).reshape(B, N_birds, -1)
            spatial_emb = F.normalize(
                self.spatial_fuse(torch.cat([spatial_emb, radio_emb], dim=-1)), dim=-1
            )

        # Similarity scores: dot product (both L2-normalized -> cosine similarity)
        # audio_emb: (B, d_embed), spatial_emb: (B, N, d_embed)
        logits = torch.bmm(spatial_emb, audio_emb.unsqueeze(-1)).squeeze(-1)  # (B, N)
        logits = logits / self.temperature

        # Mask
        if "bird_mask" in batch:
            mask = batch["bird_mask"]
            nb = mask.shape[1]
            if nb != N_birds:
                if nb < N_birds:
                    logits = logits[:, :nb]
                else:
                    pad = torch.full((B, nb - N_birds), float("-inf"), device=logits.device)
                    logits = torch.cat([logits, pad], dim=1)
            logits = logits.masked_fill(~mask, float("-inf"))

        # Localization
        pred_pos = self.loc_head(audio_emb)

        return {
            "logits": logits,
            "pred_position": pred_pos,
            "attn_weights": F.softmax(logits, dim=-1),
        }
