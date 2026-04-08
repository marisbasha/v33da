"""NeuralSRP — Physics-grounded candidate scoring with TDOA priors and RFN."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..data.dataset import MIC_POSITIONS, CAGE_X, CAGE_Y, CAGE_Z
from ..data.features import MelSpecExtractor

SPEED_OF_SOUND = 343000.0  # mm/s


class RelaxedFreqNorm(nn.Module):
    """Relaxed Instance Frequency-wise Normalization.

    Normalizes along frequency axis per-instance, with learnable relaxation
    parameter controlling how much normalization to apply.
    Removes room-specific frequency coloring while preserving discriminative info.
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, n_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, n_channels, 1, 1))
        # Relaxation parameter — 0 = full normalization, 1 = no normalization
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, F, T) -> (B, C, F, T)"""
        # Per-instance frequency statistics
        mu = x.mean(dim=2, keepdim=True)
        sigma = x.std(dim=2, keepdim=True) + 1e-6
        x_norm = (x - mu) / sigma
        # Relaxed: blend between normalized and original
        alpha = torch.sigmoid(self.alpha)
        return (alpha * x + (1 - alpha) * x_norm) * self.gamma + self.beta


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class ResBlock(nn.Module):
    """ResNet block with SE attention."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return F.relu(out)


class NeuralSRP(nn.Module):
    """
    Physics-grounded candidate scoring model.

    1. SALSA-Lite features (spectrograms + NIPDs) with RFN normalization
    2. ResNet-SE encoder for audio
    3. TDOA-based spatial priors per candidate (physics-grounded, generalizable)
    4. Per-candidate scoring via dot product + MLP
    5. Optional: gradient reversal for experiment-invariant features
    """

    def __init__(
        self,
        n_mics: int = 5,
        sample_rate: int = 24414,
        n_fft: int = 512,
        hop_length: int = 128,
        d_model: int = 256,
        dropout: float = 0.1,
        use_radio: bool = True,
        n_radio_channels: int = 7,
    ):
        super().__init__()
        self.n_mics = n_mics
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.n_freq = n_fft // 2 + 1
        self.use_radio = use_radio

        # Mic positions
        self.register_buffer('mic_pos', torch.tensor(MIC_POSITIONS, dtype=torch.float32))
        bounds = torch.tensor([CAGE_X, CAGE_Y, CAGE_Z], dtype=torch.float32)
        self.register_buffer('cage_lo', bounds[:, 0])
        self.register_buffer('cage_hi', bounds[:, 1])

        # Input channels: n_mics spectrograms + (n_mics-1) NIPDs
        n_input_ch = n_mics + (n_mics - 1)  # 5 + 4 = 9

        # RFN at input
        self.input_rfn = RelaxedFreqNorm(n_input_ch)

        # ResNet encoder with SE blocks + RFN
        self.encoder = nn.Sequential(
            ResBlock(n_input_ch, 64, stride=(2, 1)),
            RelaxedFreqNorm(64),
            ResBlock(64, 128, stride=(2, 1)),
            RelaxedFreqNorm(128),
            ResBlock(128, 256, stride=(2, 1)),
            RelaxedFreqNorm(256),
            ResBlock(256, 256, stride=(2, 1)),
        )
        # After 4x stride-2 on freq: n_freq/16
        enc_freq = self.n_freq // 16
        enc_dim = 256 * max(enc_freq, 1)

        # Temporal pooling
        self.pool = nn.AdaptiveAvgPool2d((1, None))  # pool freq, keep time
        self.audio_proj = nn.Linear(256, d_model)

        # TDOA-based spatial prior encoder
        # TDOA vector per candidate: (n_mics - 1,) relative delays
        tdoa_dim = n_mics - 1  # 4
        self.spatial_encoder = nn.Sequential(
            nn.Linear(tdoa_dim, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
        )

        # Radio temporal encoder (optional)
        if use_radio:
            self.radio_cnn = nn.Sequential(
                nn.Conv1d(n_radio_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.radio_proj = nn.Linear(64, d_model)
            scorer_in = d_model * 3  # audio + spatial + radio
        else:
            self.radio_cnn = None
            scorer_in = d_model * 2  # audio + spatial

        # Per-candidate scorer
        self.scorer = nn.Sequential(
            nn.Linear(scorer_in, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        # Localization head (from audio)
        self.loc_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3),
            nn.Sigmoid(),
        )

    def _compute_salsa_lite(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute SALSA-Lite features: spectrograms + NIPDs.

        Args:
            audio: (B, n_mics, N_samples)

        Returns:
            (B, n_mics + n_mics-1, F, T) SALSA-Lite features
        """
        B = audio.shape[0]
        window = torch.hann_window(self.n_fft, device=audio.device)

        # STFT per mic
        stfts = []
        for mi in range(self.n_mics):
            s = torch.stft(audio[:, mi], self.n_fft, self.hop_length,
                          window=window, return_complex=True)  # (B, F, T)
            stfts.append(s)

        # Log-magnitude spectrograms
        specs = [torch.log1p(s.abs()) for s in stfts]

        # NIPDs relative to mic 0
        nipds = []
        ref = stfts[0]
        for mi in range(1, self.n_mics):
            # Normalized inter-channel phase difference
            ipd = torch.angle(stfts[mi] * ref.conj())  # (B, F, T)
            nipds.append(ipd)

        # Stack: (B, 9, F, T)
        features = torch.stack(specs + nipds, dim=1)
        return features

    def _compute_tdoa(self, bird_pos_norm: torch.Tensor) -> torch.Tensor:
        """Compute expected TDOA vectors for each candidate.

        Args:
            bird_pos_norm: (B, N_birds, 3) normalized [0,1]

        Returns:
            (B, N_birds, n_mics-1) TDOA vectors in seconds
        """
        # Denormalize
        pos_mm = bird_pos_norm * (self.cage_hi - self.cage_lo) + self.cage_lo

        # Distance to each mic
        dists = torch.cdist(pos_mm, self.mic_pos.unsqueeze(0).expand(pos_mm.shape[0], -1, -1))
        # (B, N_birds, n_mics)

        # TDOA relative to mic 0
        tdoa = (dists[:, :, 1:] - dists[:, :, :1]) / SPEED_OF_SOUND
        # (B, N_birds, n_mics-1)

        return tdoa

    def forward(self, batch: dict) -> dict:
        audio = batch["audio"]  # (B, n_mics, N)
        B = audio.shape[0]

        # 1. SALSA-Lite features
        salsa = self._compute_salsa_lite(audio)  # (B, 9, F, T)

        # 2. RFN + ResNet encoder
        salsa = self.input_rfn(salsa)
        enc = self.encoder(salsa)  # (B, 256, F', T')

        # Pool frequency, keep time
        enc = self.pool(enc).squeeze(2)  # (B, 256, T')

        # Global audio embedding
        audio_emb = enc.mean(dim=-1)  # (B, 256)
        audio_emb = self.audio_proj(audio_emb)  # (B, d_model)

        # 3. TDOA spatial priors
        tdoa = self._compute_tdoa(batch["bird_positions"])  # (B, N, 4)
        N_birds = tdoa.shape[1]
        spatial_emb = self.spatial_encoder(tdoa)  # (B, N, d_model)

        # 4. Radio temporal encoding (optional)
        if self.use_radio and "bird_radio_temporal" in batch:
            rt = batch["bird_radio_temporal"]  # (B, N, n_ch, n_frames)
            rt_flat = rt.reshape(B * N_birds, rt.shape[2], rt.shape[3])
            radio_feat = self.radio_cnn(rt_flat).squeeze(-1)  # (B*N, 64)
            radio_emb = self.radio_proj(radio_feat).reshape(B, N_birds, -1)  # (B, N, d_model)
        else:
            radio_emb = None

        # 5. Per-candidate scoring
        audio_expanded = audio_emb.unsqueeze(1).expand(-1, N_birds, -1)  # (B, N, d_model)

        if radio_emb is not None:
            scorer_input = torch.cat([audio_expanded, spatial_emb, radio_emb], dim=-1)
        else:
            scorer_input = torch.cat([audio_expanded, spatial_emb], dim=-1)

        logits = self.scorer(scorer_input).squeeze(-1)  # (B, N)

        # Mask invalid birds
        if "bird_mask" in batch:
            logits = logits.masked_fill(~batch["bird_mask"], float("-inf"))

        # Localization
        attn = F.softmax(logits, dim=-1)
        pred_pos = (attn.unsqueeze(-1) * batch["bird_positions"]).sum(dim=1)

        return {
            "logits": logits,
            "pred_position": pred_pos,
            "attn_weights": attn,
        }
