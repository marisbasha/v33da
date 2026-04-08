"""Candidate branch: per-bird features → query embeddings."""

import torch
import torch.nn as nn


class CandidateEncoder(nn.Module):
    """
    Encode per-bird features into query embeddings for the decoder.

    Modality configs determine input features:
    - audio_only: pos_3d only (3D)
    - audio_3d: pos_3d + head_orient (6D)
    - audio_3d_radio: pos_3d + head_orient + radio_rssi (11D)
    """

    def __init__(self, modality: str = "audio_3d_radio", d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.modality = modality

        if modality == "audio_only":
            d_in = 3  # position only
        elif modality == "audio_3d":
            d_in = 6  # position + head orientation
        elif modality == "audio_3d_radio":
            d_in = 18  # position(3) + head(3) + radio(12)
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # Batch normalization to handle different feature scales
        self.input_norm = nn.BatchNorm1d(d_in)

        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, bird_positions: torch.Tensor, bird_head_orient: torch.Tensor,
                bird_radio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bird_positions: (B, N_birds, 3)
            bird_head_orient: (B, N_birds, 3)
            bird_radio: (B, N_birds, 5)

        Returns:
            (B, N_birds, d_model) query embeddings
        """
        if self.modality == "audio_only":
            features = bird_positions
        elif self.modality == "audio_3d":
            features = torch.cat([bird_positions, bird_head_orient], dim=-1)
        elif self.modality == "audio_3d_radio":
            features = torch.cat([bird_positions, bird_head_orient, bird_radio], dim=-1)

        # BatchNorm expects (B, C) or (B, C, L), reshape for (B, N_birds, D)
        B, N, D = features.shape
        features = self.input_norm(features.reshape(B * N, D)).reshape(B, N, D)

        return self.mlp(features)
