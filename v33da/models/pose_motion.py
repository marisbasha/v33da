"""PoseMotionCandidate: active-speaker-style candidate scoring from temporal 3D pose."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseMotionCandidate(nn.Module):
    """Score visible birds from short-window temporal pose cues only.

    Each bird receives a temporal sequence derived from 3D keypoints:
    relative keypoint coordinates plus frame-to-frame deltas.
    The model pools this sequence into a per-candidate motion embedding,
    then scores the visible candidates without using audio or radio.
    """

    def __init__(self, d_in: int = 30, hidden: int = 128):
        super().__init__()
        self.frame_proj = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.temporal = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.candidate_mlp = nn.Sequential(
            nn.Linear(hidden * 2 + 6, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pose = batch["bird_pose_temporal"]  # (B, N, T, D)
        pos = batch["bird_positions"]
        head = batch["bird_head_orient"]
        bird_mask = batch["bird_mask"]
        frame_mask = batch["pose_frame_mask"]  # (B, T)

        B, N, T, D = pose.shape
        pose = pose.reshape(B * N, T, D)
        frame_feat = self.frame_proj(pose)
        seq_out, _ = self.temporal(frame_feat)

        fm = frame_mask[:, None, :, None].expand(B, N, T, 1).reshape(B * N, T, 1).to(seq_out.dtype)
        denom = fm.sum(dim=1).clamp_min(1.0)
        pooled_mean = (seq_out * fm).sum(dim=1) / denom

        # Also keep a max-pooled burst feature for short beak/head motion spikes.
        neg_inf = torch.full_like(seq_out, -1e9)
        pooled_max = torch.where(fm > 0, seq_out, neg_inf).max(dim=1).values
        pooled = 0.5 * (pooled_mean + pooled_max)
        pooled = pooled.reshape(B, N, -1)

        cand_feat = torch.cat([pooled, pos, head], dim=-1)
        logits = self.candidate_mlp(cand_feat).squeeze(-1)
        logits = logits.masked_fill(~bird_mask, float("-inf"))
        attn = F.softmax(logits, dim=-1)
        pred_pos = (attn.unsqueeze(-1) * pos).sum(dim=1)
        return {"logits": logits, "pred_position": pred_pos}

