"""VideoCandidate: top/back RGB candidate scorer from per-bird crop tubes."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoCandidate(nn.Module):
    """Active-speaker-style candidate scorer from short top/back video crops."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.view_fuser = nn.Sequential(
            nn.Linear(96 * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.temporal = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.candidate_head = nn.Sequential(
            nn.Linear(hidden * 2 + 6, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        video = batch["bird_video_crops"]  # (B, N, T, V, C, H, W)
        pos = batch["bird_positions"]
        head = batch["bird_head_orient"]
        bird_mask = batch["bird_mask"]
        frame_mask = batch["video_frame_mask"]  # (B, T)

        B, N, T, V, C, H, W = video.shape
        x = video.reshape(B * N * T * V, C, H, W)
        feat = self.frame_encoder(x).flatten(1)
        feat = feat.reshape(B * N * T, V * feat.shape[-1])
        feat = self.view_fuser(feat).reshape(B * N, T, -1)

        seq_out, _ = self.temporal(feat)
        fm = frame_mask[:, None, :, None].expand(B, N, T, 1).reshape(B * N, T, 1).to(seq_out.dtype)
        denom = fm.sum(dim=1).clamp_min(1.0)
        pooled_mean = (seq_out * fm).sum(dim=1) / denom
        neg_inf = torch.full_like(seq_out, -1e9)
        pooled_max = torch.where(fm > 0, seq_out, neg_inf).max(dim=1).values
        pooled = 0.5 * (pooled_mean + pooled_max)
        pooled = pooled.reshape(B, N, -1)

        cand_feat = torch.cat([pooled, pos, head], dim=-1)
        logits = self.candidate_head(cand_feat).squeeze(-1)
        logits = logits.masked_fill(~bird_mask, float("-inf"))
        attn = F.softmax(logits, dim=-1)
        pred_position = (attn.unsqueeze(-1) * pos).sum(dim=1)
        return {"logits": logits, "pred_position": pred_position, "attn_weights": attn}
