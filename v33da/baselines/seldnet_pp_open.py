"""SELDnet++ Open-Set — candidate-only scoring, no fixed identity head.

Uses: 5-ch audio + 3D positions + head orientation + radio (summary + temporal).
Attribution: purely candidate-conditioned (works on unseen birds).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.features import MelSpecExtractor


class SELDnetPPOpen(nn.Module):
    """
    SELDnet++ without fixed identity head — purely candidate-conditioned.

    Audio path: mel-spec → CNN → BiGRU → audio embedding (256)
    Candidate path: [3D pos + head orient + radio summary + radio temporal enc] → MLP → 128D
    Scoring: normalized dot product between audio embedding and each candidate embedding.
    No direct classification head — works on any number of candidates including unseen birds.
    """

    def __init__(
        self,
        n_mics: int = 5,
        sample_rate: int = 24414,
        n_mels: int = 80,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_mics = n_mics
        self.mel = MelSpecExtractor(sample_rate=sample_rate, n_mels=n_mels)

        # Audio encoder
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
        gru_dim = hidden_dim * 2  # 256

        # Radio temporal encoder
        self.radio_temporal_enc = nn.Sequential(
            nn.Conv1d(7, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Candidate encoder: 3 + 3 + 12 + 64 = 82 → 128
        candidate_input_dim = 3 + 3 + 12 + 64
        self.candidate_enc = nn.Sequential(
            nn.Linear(candidate_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
        )

        # Audio projection to candidate matching space
        self.audio_to_emb = nn.Linear(gru_dim, 128)

        # Temperature for dot product scoring
        self.temperature = nn.Parameter(torch.tensor(10.0))

        # DOA head (position prediction from audio)
        self.doa_head = nn.Sequential(
            nn.Linear(gru_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

    def encode_audio(self, audio):
        """Encode audio to pooled embedding. Mel on CPU for STFT compat."""
        device = audio.device
        audio_cpu = audio.cpu()
        mels = torch.stack([self.mel(audio_cpu[:, mi]) for mi in range(self.n_mics)], dim=1)
        mels = mels.to(device)
        mels = (mels - mels.mean(dim=(2, 3), keepdim=True)) / (mels.std(dim=(2, 3), keepdim=True) + 1e-8)
        x = self.cnn(mels).squeeze(2).permute(0, 2, 1)
        x, _ = self.gru(x)
        return x.mean(dim=1)

    def encode_candidates(self, batch):
        """Encode per-candidate features."""
        pos = batch["bird_positions"]
        B, N, _ = pos.shape
        device = pos.device

        head = batch.get("bird_head_orient", torch.zeros(B, N, 3, device=device))
        radio_summary = batch.get("bird_radio", torch.zeros(B, N, 12, device=device))

        radio_temporal = batch.get("bird_radio_temporal", None)
        if radio_temporal is not None and radio_temporal.shape[-1] > 1:
            rt = radio_temporal.reshape(B * N, radio_temporal.shape[2], radio_temporal.shape[3])
            rt_enc = self.radio_temporal_enc(rt).squeeze(-1).reshape(B, N, 64)
        else:
            rt_enc = torch.zeros(B, N, 64, device=device)

        cand_feat = torch.cat([pos, head, radio_summary, rt_enc], dim=-1)
        return self.candidate_enc(cand_feat)

    def forward(self, batch: dict) -> dict:
        # Handle cached mels or raw audio
        if "cached_mel" in batch:
            mels = batch["cached_mel"]
            x = self.cnn(mels).squeeze(2).permute(0, 2, 1)
            x, _ = self.gru(x)
            audio_emb = x.mean(dim=1)
        else:
            audio_emb = self.encode_audio(batch["audio"])

        cand_emb = self.encode_candidates(batch)
        N = cand_emb.shape[1]

        # Pure candidate matching — no direct head
        audio_proj = F.normalize(self.audio_to_emb(audio_emb), dim=-1)
        cand_norm = F.normalize(cand_emb, dim=-1)
        logits = torch.bmm(cand_norm, audio_proj.unsqueeze(-1)).squeeze(-1) * self.temperature

        if "bird_mask" in batch:
            logits = logits.masked_fill(~batch["bird_mask"], float("-inf"))

        position = self.doa_head(audio_emb)
        attn = F.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "pred_position": position,
            "attn_weights": attn,
        }
