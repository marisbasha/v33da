"""SELDnet++ — multimodal extension using all available modalities.

Uses: 5-ch audio + 3D positions + head orientation + radio (summary + temporal).
Attribution: candidate-conditioned scoring (not fixed identity head).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.features import MelSpecExtractor


class SELDnetPP(nn.Module):
    """
    SELDnet++ with full multimodal fusion.

    Audio path: mel-spec → CNN → BiGRU → audio embedding (256)
    Candidate path: [3D pos (3) + head orient (3) + radio summary (12) + radio temporal enc (64)] → MLP → candidate embedding (128)
    Scoring: audio embedding ⊙ candidate embedding → per-candidate score
    Also retains a direct classification head blended via learnable β.
    """

    def __init__(
        self,
        n_mics: int = 5,
        sample_rate: int = 24414,
        n_mels: int = 80,
        max_birds: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_mics = n_mics
        self.max_birds = max_birds
        self.mel = MelSpecExtractor(sample_rate=sample_rate, n_mels=n_mels)

        # Audio encoder: CNN → BiGRU
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

        # Radio temporal encoder: Conv1d per bird
        self.radio_temporal_enc = nn.Sequential(
            nn.Conv1d(7, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # → (B*N, 64, 1)
        )

        # Candidate encoder: static features (18D) + radio temporal (64D) → 128D
        # 3 (pos) + 3 (head) + 12 (radio summary) + 64 (radio temporal) = 82
        candidate_input_dim = 3 + 3 + 12 + 64
        self.candidate_enc = nn.Sequential(
            nn.Linear(candidate_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
        )

        # Audio projection for candidate matching
        self.audio_to_emb = nn.Linear(gru_dim, 128)

        # Direct classification head (for blending)
        self.direct_head = nn.Sequential(
            nn.Linear(gru_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_birds),
        )

        # Learnable blend parameter
        self.beta_logit = nn.Parameter(torch.tensor(0.5))

        # DOA head
        self.doa_head = nn.Sequential(
            nn.Linear(gru_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

    def encode_audio(self, audio):
        """Encode audio to a pooled embedding. Mel computed on CPU for STFT compatibility."""
        B = audio.shape[0]
        device = audio.device

        # Mel on CPU to avoid CUDA STFT kernel issues
        audio_cpu = audio.cpu()
        mels = torch.stack([self.mel(audio_cpu[:, mi]) for mi in range(self.n_mics)], dim=1)
        mels = mels.to(device)

        # Instance normalization
        mels = (mels - mels.mean(dim=(2, 3), keepdim=True)) / (mels.std(dim=(2, 3), keepdim=True) + 1e-8)

        # CNN → GRU → pool
        x = self.cnn(mels)  # (B, 256, 1, T)
        x = x.squeeze(2).permute(0, 2, 1)  # (B, T, 256)
        x, _ = self.gru(x)  # (B, T, 256)
        pooled = x.mean(dim=1)  # (B, 256)
        return pooled

    def encode_candidates(self, batch):
        """Encode per-candidate features using all available modalities."""
        pos = batch["bird_positions"]  # (B, N, 3)
        B, N, _ = pos.shape
        device = pos.device

        head = batch.get("bird_head_orient", torch.zeros(B, N, 3, device=device))
        radio_summary = batch.get("bird_radio", torch.zeros(B, N, 12, device=device))

        # Radio temporal encoding
        radio_temporal = batch.get("bird_radio_temporal", None)
        if radio_temporal is not None and radio_temporal.shape[-1] > 1:
            # (B, N, 7, T) → (B*N, 7, T)
            rt = radio_temporal.reshape(B * N, radio_temporal.shape[2], radio_temporal.shape[3])
            rt_enc = self.radio_temporal_enc(rt).squeeze(-1)  # (B*N, 64)
            rt_enc = rt_enc.reshape(B, N, 64)
        else:
            rt_enc = torch.zeros(B, N, 64, device=device)

        # Concatenate all candidate features
        cand_feat = torch.cat([pos, head, radio_summary, rt_enc], dim=-1)  # (B, N, 82)
        cand_emb = self.candidate_enc(cand_feat)  # (B, N, 128)
        return cand_emb

    def forward(self, batch: dict) -> dict:
        audio = batch["audio"]
        B = audio.shape[0]

        # Encode audio
        audio_emb = self.encode_audio(audio)  # (B, 256)

        # Encode candidates
        cand_emb = self.encode_candidates(batch)  # (B, N, 128)
        N = cand_emb.shape[1]

        # Candidate-conditioned scoring: dot product
        audio_proj = F.normalize(self.audio_to_emb(audio_emb), dim=-1)  # (B, 128)
        cand_norm = F.normalize(cand_emb, dim=-1)  # (B, N, 128)
        dot_logits = torch.bmm(cand_norm, audio_proj.unsqueeze(-1)).squeeze(-1) * 10  # (B, N)

        # Direct head (fixed identity)
        direct_logits = self.direct_head(audio_emb)  # (B, max_birds)
        if N < self.max_birds:
            direct_logits = direct_logits[:, :N]
        elif N > self.max_birds:
            pad = torch.full((B, N - self.max_birds), float("-inf"), device=audio.device)
            direct_logits = torch.cat([direct_logits, pad], dim=1)

        # Blend
        beta = torch.sigmoid(self.beta_logit)
        logits = (1 - beta) * direct_logits + beta * dot_logits

        # Mask invalid birds
        if "bird_mask" in batch:
            logits = logits.masked_fill(~batch["bird_mask"], float("-inf"))

        # Position prediction
        position = self.doa_head(audio_emb)

        attn = F.softmax(logits, dim=-1)
        return {
            "logits": logits,
            "pred_position": position,
            "attn_weights": attn,
            "beta": beta.detach(),
        }
