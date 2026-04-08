"""PairRank: pairwise candidate ranking without a closed-set class head."""

from __future__ import annotations

from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.dataset import CAGE_X, CAGE_Y, CAGE_Z, MIC_POSITIONS


SPEED_OF_SOUND = 343000.0
SR = 24414.0625
MIC_PAIRS = list(combinations(range(5), 2))


class PairRankNet(nn.Module):
    """Candidate scorer built from pairwise preferences.

    For each candidate pair (i, j), predict whether i is more plausible than j.
    Aggregate pairwise wins into per-candidate scores.
    """

    def __init__(self, d_model: int = 96, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("mic_pos", torch.tensor(MIC_POSITIONS, dtype=torch.float32))
        bounds = torch.tensor([CAGE_X, CAGE_Y, CAGE_Z], dtype=torch.float32)
        self.register_buffer("cage_lo", bounds[:, 0])
        self.register_buffer("cage_hi", bounds[:, 1])

        self.audio_encoder = nn.Sequential(
            nn.Linear(len(MIC_PAIRS) * 3, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.cand_encoder = nn.Sequential(
            nn.Linear(3 + 3 + 12 + len(MIC_PAIRS), d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.pair_scorer = nn.Sequential(
            nn.Linear(d_model * 2 + 5, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def _denorm(self, pos):
        return pos * (self.cage_hi - self.cage_lo) + self.cage_lo

    def _gcc_stats(self, audio: torch.Tensor, pos: torch.Tensor):
        """Return audio summary and expected-delay features."""
        bsz, _, n_samples = audio.shape
        audio = (audio - audio.mean(dim=-1, keepdim=True)) / (audio.std(dim=-1, keepdim=True) + 1e-8)
        spectra = [torch.fft.rfft(audio[:, mi], dim=-1) for mi in range(audio.shape[1])]

        audio_stats = []
        per_candidate = []
        pos_mm = self._denorm(pos)
        center = n_samples // 2

        for mi, mj in MIC_PAIRS:
            cross = spectra[mi] * torch.conj(spectra[mj])
            cc = torch.fft.irfft(cross / (cross.abs() + 1e-10), n=n_samples, dim=-1)
            cc = torch.cat([cc[:, n_samples // 2 :], cc[:, : n_samples // 2 + 1]], dim=-1)

            peak = cc.max(dim=-1).values
            mean = cc.mean(dim=-1)
            sharp = peak / (mean.abs() + 1e-6)
            audio_stats.append(torch.stack([peak, mean, sharp], dim=-1))

            d_i = torch.linalg.norm(pos_mm - self.mic_pos[mi], dim=-1)
            d_j = torch.linalg.norm(pos_mm - self.mic_pos[mj], dim=-1)
            expected = (d_i - d_j) / SPEED_OF_SOUND * SR
            per_candidate.append(expected / 50.0)

        audio_stats = torch.cat(audio_stats, dim=-1)
        per_candidate = torch.stack(per_candidate, dim=-1)
        return audio_stats, per_candidate

    def forward(self, batch):
        pos = batch["bird_positions"]
        head = batch["bird_head_orient"]
        radio = batch["bird_radio"]
        mask = batch["bird_mask"]

        audio_stats, expected = self._gcc_stats(batch["audio"], pos)
        audio_emb = self.audio_encoder(audio_stats)

        cand_feat = torch.cat([pos, head, radio, expected], dim=-1)
        cand_emb = self.cand_encoder(cand_feat)

        bsz, n_birds, d_model = cand_emb.shape
        scores = torch.zeros(bsz, n_birds, device=pos.device)

        for i in range(n_birds):
            for j in range(i + 1, n_birds):
                rel = pos[:, i] - pos[:, j]
                rel_dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
                # Difference in expected TDOA summary acts as a geometry cue.
                tdoa_diff = (expected[:, i] - expected[:, j]).mean(dim=-1, keepdim=True)
                pair_in = torch.cat([cand_emb[:, i], cand_emb[:, j], rel, rel_dist, tdoa_diff], dim=-1)
                pref_ij = self.pair_scorer(pair_in).squeeze(-1)
                scores[:, i] += pref_ij
                scores[:, j] -= pref_ij

        scores = scores + 0.05 * (cand_emb * audio_emb.unsqueeze(1)).sum(dim=-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores.detach(), dim=-1)
        pred_pos = (attn.unsqueeze(-1) * pos).sum(dim=1)
        return {"logits": scores, "pred_position": pred_pos}


def pairrank_loss(outputs, batch):
    logits = outputs["logits"]
    labels = batch["label_idx"]
    ce = F.cross_entropy(logits, labels, label_smoothing=0.03)
    loc = F.mse_loss(outputs["pred_position"], batch["label_position"])

    pair_loss = 0.0
    count = 0
    for i in range(logits.shape[1]):
        for j in range(logits.shape[1]):
            if i == j:
                continue
            target = ((labels == i) & (labels != j)).float()
            valid = batch["bird_mask"][:, i] & batch["bird_mask"][:, j]
            if valid.any():
                diff = logits[valid, i] - logits[valid, j]
                targ = target[valid]
                pair_loss = pair_loss + F.binary_cross_entropy_with_logits(diff, targ)
                count += 1
    if count > 0:
        pair_loss = pair_loss / count
    loss = ce + 0.25 * pair_loss + 0.05 * loc
    return loss, {"ce": float(ce.detach()), "pair": float(pair_loss.detach()) if count > 0 else 0.0}
