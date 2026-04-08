"""Small Conformer encoder for the spatial stream."""

import torch
import torch.nn as nn
import math


class ConformerBlock(nn.Module):
    """Single Conformer block: FFN → MHSA → Conv → FFN."""

    def __init__(self, d_model=256, n_heads=4, ff_mult=4, conv_kernel=31, dropout=0.1):
        super().__init__()
        # First FFN (half-step)
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * ff_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )
        # Multi-head self-attention
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        # Depthwise convolution
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, conv_kernel, padding=conv_kernel // 2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, 1),
            nn.Dropout(dropout),
        )
        # Second FFN (half-step)
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * ff_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, T, D)"""
        # FFN half-step
        x = x + 0.5 * self.ffn1(x)
        # Self-attention
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.attn_dropout(attn_out)
        # Convolution
        x_norm = self.conv_norm(x)
        x_conv = self.conv(x_norm.transpose(1, 2)).transpose(1, 2)
        x = x + x_conv
        # FFN half-step
        x = x + 0.5 * self.ffn2(x)
        x = self.final_norm(x)
        return x


class SpatialConformer(nn.Module):
    """
    Conformer encoder for SALSA-Lite spatial features.

    Input: (B, C, T, F) SALSA features where C=n_mics, F=n_mels
    Output: (B, T, d_model) spatial embedding sequence
    """

    def __init__(self, n_input_channels=7, n_mels=80, d_model=256, n_layers=2,
                 n_heads=4, dropout=0.1):
        super().__init__()
        # Project SALSA features: (C * F) -> d_model
        self.input_proj = nn.Linear(n_input_channels * n_mels, d_model)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, salsa: torch.Tensor) -> torch.Tensor:
        """(B, C, T, F) -> (B, T, d_model)"""
        B, C, T, F = salsa.shape
        # Flatten channel and frequency: (B, T, C*F)
        x = salsa.permute(0, 2, 1, 3).reshape(B, T, C * F)
        x = self.input_proj(x)  # (B, T, d_model)
        for block in self.blocks:
            x = block(x)
        return x
