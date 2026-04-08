"""Reviewer-facing neural baselines adapted from the SELD literature."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.dataset import CAGE_X, CAGE_Y, CAGE_Z
from ..data.features import SALSALiteExtractor
from .candidate_temporal import JointRadioTemporalEncoder
from .spatial.conformer import SpatialConformer


def _bounds(device=None):
    return torch.tensor([CAGE_X, CAGE_Y, CAGE_Z], dtype=torch.float32, device=device)


class _CandidateScorer(nn.Module):
    def __init__(self, audio_dim: int, radio_mode: str = "summary", radio_context_dim: int = 64):
        super().__init__()
        self.radio_mode = radio_mode
        self.radio_encoder = None
        bird_dim = 3 + 3 + 12
        if radio_mode == "joint_temporal":
            self.radio_encoder = JointRadioTemporalEncoder(d_out=radio_context_dim)
            bird_dim += 2 * radio_context_dim
        self.net = nn.Sequential(
            nn.Linear(audio_dim + bird_dim, audio_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(audio_dim, audio_dim // 2),
            nn.ReLU(),
            nn.Linear(audio_dim // 2, 1),
        )

    def forward(self, audio_emb: torch.Tensor, batch: dict) -> torch.Tensor:
        bird_feat = [batch["bird_positions"], batch["bird_head_orient"], batch["bird_radio"]]
        if self.radio_encoder is not None:
            bird_radio_ctx, global_radio_ctx = self.radio_encoder(
                batch["bird_radio_temporal"],
                batch["bird_mask"],
            )
            bird_feat.append(bird_radio_ctx)
            bird_feat.append(global_radio_ctx.unsqueeze(1).expand(-1, batch["bird_positions"].shape[1], -1))
        bird_feat = torch.cat(bird_feat, dim=-1)
        audio_expand = audio_emb.unsqueeze(1).expand(-1, bird_feat.shape[1], -1)
        logits = self.net(torch.cat([audio_expand, bird_feat], dim=-1)).squeeze(-1)
        return logits.masked_fill(~batch["bird_mask"], float("-inf"))


class _PositionBiasScorer(nn.Module):
    def __init__(self, hidden_dim: int = 96, radio_context_dim: int = 64):
        super().__init__()
        self.radio_encoder = JointRadioTemporalEncoder(d_out=radio_context_dim)
        in_dim = 3 + 3 + 12 + 3 + 3 + 1 + 2 * radio_context_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pred_position: torch.Tensor, batch: dict) -> torch.Tensor:
        pos = batch["bird_positions"]
        delta = pos - pred_position.unsqueeze(1)
        dist = torch.norm(delta, dim=-1, keepdim=True)
        bird_radio_ctx, global_radio_ctx = self.radio_encoder(
            batch["bird_radio_temporal"],
            batch["bird_mask"],
        )
        feat = torch.cat(
            [
                pos,
                batch["bird_head_orient"],
                batch["bird_radio"],
                pred_position.unsqueeze(1).expand(-1, pos.shape[1], -1),
                delta,
                dist,
                bird_radio_ctx,
                global_radio_ctx.unsqueeze(1).expand(-1, pos.shape[1], -1),
            ],
            dim=-1,
        )
        logits = self.net(feat).squeeze(-1)
        return logits.masked_fill(~batch["bird_mask"], float("-inf"))


class _SpatialEncoder(nn.Module):
    def __init__(self, n_mics: int = 5, n_mels: int = 64, d_model: int = 192):
        super().__init__()
        self.salsa = SALSALiteExtractor(n_mics=n_mics, n_mels=n_mels)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_mics, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(128, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.AdaptiveAvgPool2d((None, 1)),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.salsa(audio)  # (B, C, T, F)
        x = self.cnn(x).squeeze(-1)  # (B, D, T)
        return x.transpose(1, 2)  # (B, T, D)


class AccdoaAttribution(nn.Module):
    """ACCDOA-style CRNN adapted to candidate attribution."""

    def __init__(self, n_mics: int = 5, n_mels: int = 64, d_model: int = 192, radio_mode: str = "none"):
        super().__init__()
        self.encoder = _SpatialEncoder(n_mics=n_mics, n_mels=n_mels, d_model=d_model)
        self.gru = nn.GRU(
            d_model,
            d_model // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.vec_head = nn.Linear(d_model, 3)
        self.activity_head = nn.Linear(d_model, 1)
        self.distance_scale = nn.Parameter(torch.tensor(12.0))
        self.radio_bias = _PositionBiasScorer() if radio_mode == "joint_temporal" else None
        self.radio_bias_scale = nn.Parameter(torch.tensor(0.5)) if self.radio_bias is not None else None

    def forward(self, batch: dict) -> dict:
        seq = self.encoder(batch["audio"])
        seq, _ = self.gru(seq)
        frame_activity = torch.sigmoid(self.activity_head(seq)).squeeze(-1)
        weights = frame_activity / frame_activity.sum(dim=1, keepdim=True).clamp_min(1e-6)
        frame_pos = torch.sigmoid(self.vec_head(seq))
        pred_position = torch.sum(weights.unsqueeze(-1) * frame_pos, dim=1)

        dists = torch.norm(batch["bird_positions"] - pred_position.unsqueeze(1), dim=-1)
        logits = -F.softplus(self.distance_scale) * dists
        if self.radio_bias is not None:
            logits = logits + self.radio_bias_scale * torch.tanh(self.radio_bias(pred_position, batch))
        logits = logits.masked_fill(~batch["bird_mask"], float("-inf"))
        attn = F.softmax(logits, dim=-1)
        return {
            "logits": logits,
            "pred_position": pred_position,
            "attn_weights": attn,
            "frame_activity": frame_activity,
        }


class CSTFormerAttribution(nn.Module):
    """CST-former-inspired transformer baseline with candidate scoring."""

    def __init__(self, n_mics: int = 5, n_mels: int = 64, d_model: int = 192, radio_mode: str = "summary"):
        super().__init__()
        self.salsa = SALSALiteExtractor(n_mics=n_mics, n_mels=n_mels)
        self.encoder = SpatialConformer(
            n_input_channels=n_mics,
            n_mels=n_mels,
            d_model=d_model,
            n_layers=3,
            n_heads=4,
            dropout=0.1,
        )
        self.pool = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
        )
        self.pos_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3),
            nn.Sigmoid(),
        )
        self.scorer = _CandidateScorer(d_model, radio_mode=radio_mode)

    def forward(self, batch: dict) -> dict:
        salsa = self.salsa(batch["audio"])
        seq = self.encoder(salsa)
        audio_emb = self.pool(seq.mean(dim=1))
        pred_position = self.pos_head(audio_emb)
        logits = self.scorer(audio_emb, batch)
        attn = F.softmax(logits, dim=-1)
        return {
            "logits": logits,
            "pred_position": pred_position,
            "attn_weights": attn,
        }


class EINV2Attribution(nn.Module):
    """Trackwise event-independent baseline adapted to attribution."""

    def __init__(self, n_tracks: int = 3, n_mics: int = 5, n_mels: int = 64, d_model: int = 192, radio_mode: str = "none"):
        super().__init__()
        self.n_tracks = n_tracks
        self.encoder = _SpatialEncoder(n_mics=n_mics, n_mels=n_mels, d_model=d_model)
        self.gru = nn.GRU(
            d_model,
            d_model // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.track_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.track_activity = nn.Linear(d_model, n_tracks)
        self.track_pos = nn.Linear(d_model, n_tracks * 3)
        self.distance_scale = nn.Parameter(torch.tensor(12.0))
        self.radio_scorer = _CandidateScorer(d_model, radio_mode="joint_temporal") if radio_mode == "joint_temporal" else None
        self.radio_bias_scale = nn.Parameter(torch.tensor(0.5)) if self.radio_scorer is not None else None

    def forward(self, batch: dict) -> dict:
        seq = self.encoder(batch["audio"])
        seq, _ = self.gru(seq)
        pooled = self.track_proj(seq.mean(dim=1))
        track_logits = self.track_activity(pooled)
        track_pos = torch.sigmoid(self.track_pos(pooled)).view(-1, self.n_tracks, 3)

        dists = torch.cdist(track_pos, batch["bird_positions"])
        candidate_track_logits = track_logits.unsqueeze(-1) - F.softplus(self.distance_scale) * dists
        mask = batch["bird_mask"].unsqueeze(1)
        candidate_track_logits = candidate_track_logits.masked_fill(~mask, float("-inf"))
        logits = candidate_track_logits.max(dim=1).values
        if self.radio_scorer is not None:
            logits = logits + self.radio_bias_scale * torch.tanh(self.radio_scorer(pooled, batch))

        best_track = track_logits.argmax(dim=-1)
        pred_position = track_pos[torch.arange(track_pos.shape[0], device=track_pos.device), best_track]
        attn = F.softmax(logits, dim=-1)
        return {
            "logits": logits,
            "pred_position": pred_position,
            "attn_weights": attn,
            "track_logits": track_logits,
            "track_pos": track_pos,
        }


def literature_model(name: str) -> nn.Module:
    key = name.lower()
    if key == "accdoa":
        return AccdoaAttribution()
    if key == "accdoa_radioctx":
        return AccdoaAttribution(radio_mode="joint_temporal")
    if key == "cstformer":
        return CSTFormerAttribution()
    if key == "cstformer_radioctx":
        return CSTFormerAttribution(radio_mode="joint_temporal")
    if key == "einv2":
        return EINV2Attribution()
    if key == "einv2_radioctx":
        return EINV2Attribution(radio_mode="joint_temporal")
    raise ValueError(f"Unknown literature baseline '{name}'")


def literature_loss(name: str, outputs: dict, batch: dict, lambda_loc: float = 0.05) -> tuple[torch.Tensor, dict]:
    logits = outputs["logits"]
    ce = F.cross_entropy(logits, batch["label_idx"])
    loc = F.mse_loss(outputs["pred_position"], batch["label_position"])
    loss = ce + lambda_loc * loc
    aux = {
        "ce": float(ce.detach()),
        "loc": float(loc.detach()),
    }

    if name.lower().startswith("einv2"):
        track_logits = outputs["track_logits"]
        track_pos = outputs["track_pos"]
        true_pos = batch["label_position"].unsqueeze(1)
        track_dists = ((track_pos - true_pos) ** 2).mean(dim=-1)
        best_track = track_dists.argmin(dim=-1)
        activity_target = torch.zeros_like(track_logits)
        activity_target.scatter_(1, best_track.unsqueeze(1), 1.0)
        activity = F.binary_cross_entropy_with_logits(track_logits, activity_target)
        best_pos = track_pos[torch.arange(track_pos.shape[0], device=track_pos.device), best_track]
        track_loc = F.mse_loss(best_pos, batch["label_position"])
        loss = loss + 0.2 * activity + 0.5 * lambda_loc * track_loc
        aux["activity"] = float(activity.detach())
        aux["track_loc"] = float(track_loc.detach())

    if name.lower().startswith("accdoa"):
        frame_activity = outputs["frame_activity"]
        active_target = torch.ones_like(frame_activity)
        activity = F.binary_cross_entropy(frame_activity, active_target)
        loss = loss + 0.05 * activity
        aux["activity"] = float(activity.detach())

    return loss, aux
