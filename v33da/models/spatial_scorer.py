"""SpatialScorer — GCC-PHAT features + TDOA candidate scoring + adversarial training.

Key design principles:
1. NO mel spectrograms — prevents voice identity shortcut
2. GCC-PHAT features only — purely spatial information
3. Physics-grounded TDOA candidate scoring
4. Gradient reversal to strip experiment/bird identity
5. Channel augmentation (swap + dropout) for robustness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import combinations

from ..data.dataset import MIC_POSITIONS, CAGE_X, CAGE_Y, CAGE_Z

SPEED_OF_SOUND = 343000.0  # mm/s


class GradientReversal(torch.autograd.Function):
    """Gradient reversal layer for adversarial training."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class SpatialScorer(nn.Module):
    """
    Purely spatial model for vocal attribution.

    Input: multi-channel audio -> GCC-PHAT cross-correlations (10 mic pairs)
    Scoring: for each candidate bird, compare observed GCC pattern to expected TDOA
    Adversarial: gradient reversal strips experiment-specific features

    No mel spectrograms. No voice identity. Only spatial cues.
    """

    def __init__(
        self,
        n_mics: int = 5,
        sample_rate: int = 24414,
        n_fft: int = 512,
        max_delay: int = 60,  # max TDOA in samples (~2.5ms)
        d_model: int = 256,
        dropout: float = 0.2,
        n_experiments: int = 2,  # for adversarial head
        use_radio: bool = True,
        n_radio_channels: int = 7,
    ):
        super().__init__()
        self.n_mics = n_mics
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.max_delay = max_delay
        self.use_radio = use_radio
        self.adv_alpha = 0.0  # start with no reversal, increase during training

        # Mic pair indices
        self.mic_pairs = list(combinations(range(n_mics), 2))  # 10 pairs
        n_pairs = len(self.mic_pairs)

        # Mic positions buffer
        self.register_buffer('mic_pos', torch.tensor(MIC_POSITIONS, dtype=torch.float32))
        bounds = torch.tensor([CAGE_X, CAGE_Y, CAGE_Z], dtype=torch.float32)
        self.register_buffer('cage_lo', bounds[:, 0])
        self.register_buffer('cage_hi', bounds[:, 1])

        # GCC feature size: n_pairs x (2*max_delay+1) time lags
        gcc_feature_dim = n_pairs  # channels
        gcc_lag_dim = 2 * max_delay + 1  # "frequency" dimension

        # CNN on GCC-PHAT features: (B, n_pairs, gcc_lag_dim, T_frames)
        self.gcc_encoder = nn.Sequential(
            nn.Conv2d(n_pairs, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 1)),  # pool to fixed size
            nn.Flatten(),
            nn.Linear(128 * 8, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # TDOA encoder per candidate
        self.tdoa_encoder = nn.Sequential(
            nn.Linear(n_pairs, 128),  # TDOA for each mic pair
            nn.ReLU(),
            nn.Linear(128, d_model),
        )

        # Radio encoder (optional)
        if use_radio:
            self.radio_encoder = nn.Sequential(
                nn.Conv1d(n_radio_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(32, d_model),
                nn.ReLU(),
            )
            scorer_in = d_model * 3 + d_model  # audio + tdoa + element_product + radio
        else:
            self.radio_encoder = None
            scorer_in = d_model * 3  # audio + tdoa + element_product

        # Per-candidate scorer
        self.scorer = nn.Sequential(
            nn.Linear(scorer_in, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        # Adversarial experiment classifier (gradient reversed)
        self.adv_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, n_experiments),
        )

        # Localization head
        self.loc_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3),
            nn.Sigmoid(),
        )

    def compute_gcc_phat(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute GCC-PHAT cross-correlations for all mic pairs.

        Args:
            audio: (B, n_mics, N_samples)

        Returns:
            gcc: (B, n_pairs, 2*max_delay+1, n_frames)
        """
        B, M, N = audio.shape
        window = torch.hann_window(self.n_fft, device=audio.device)

        # STFT per mic
        stfts = []
        for mi in range(M):
            s = torch.stft(audio[:, mi], self.n_fft, self.n_fft // 2,
                          window=window, return_complex=True)
            stfts.append(s)  # (B, F, T)

        n_frames = stfts[0].shape[2]
        gcc_all = []

        for mi, mj in self.mic_pairs:
            # Cross power spectrum with phase transform
            cross = stfts[mi] * stfts[mj].conj()
            cross_norm = cross / (cross.abs() + 1e-8)  # PHAT weighting

            # Inverse FFT to get cross-correlation
            cc = torch.fft.irfft(cross_norm, dim=1)  # (B, n_fft//2+1, T) -> (B, N_cc, T)

            # Extract lags around zero
            n_cc = cc.shape[1]
            # Positive lags: 0..max_delay
            pos = cc[:, :self.max_delay + 1, :]
            # Negative lags: -max_delay..-1
            neg = cc[:, -(self.max_delay):, :]
            # Concat: [-max_delay...-1, 0...max_delay]
            gcc = torch.cat([neg, pos], dim=1)  # (B, 2*max_delay+1, T)
            gcc_all.append(gcc)

        return torch.stack(gcc_all, dim=1)  # (B, n_pairs, 2*max_delay+1, T)

    def compute_tdoa(self, bird_pos_norm: torch.Tensor) -> torch.Tensor:
        """Compute expected TDOA for each candidate at each mic pair.

        Args:
            bird_pos_norm: (B, N_birds, 3) normalized [0,1]

        Returns:
            tdoa: (B, N_birds, n_pairs) in samples
        """
        pos_mm = bird_pos_norm * (self.cage_hi - self.cage_lo) + self.cage_lo
        B, N, _ = pos_mm.shape

        tdoas = []
        for mi, mj in self.mic_pairs:
            d_i = torch.linalg.norm(pos_mm - self.mic_pos[mi], dim=-1)  # (B, N)
            d_j = torch.linalg.norm(pos_mm - self.mic_pos[mj], dim=-1)  # (B, N)
            tdoa_sec = (d_i - d_j) / SPEED_OF_SOUND
            tdoa_samples = tdoa_sec * self.sample_rate
            tdoas.append(tdoa_samples)

        return torch.stack(tdoas, dim=-1)  # (B, N, n_pairs)

    def forward(self, batch: dict) -> dict:
        audio = batch["audio"]  # (B, n_mics, N)
        B = audio.shape[0]

        # 1. GCC-PHAT features (purely spatial)
        gcc = self.compute_gcc_phat(audio)  # (B, 10, 121, T)

        # 2. Encode GCC to audio embedding
        audio_emb = self.gcc_encoder(gcc)  # (B, d_model)

        # 3. Expected TDOA per candidate
        tdoa = self.compute_tdoa(batch["bird_positions"])  # (B, N, 10)
        N_birds = tdoa.shape[1]
        # Normalize TDOA to [-1, 1] range
        tdoa_norm = tdoa / self.max_delay
        tdoa_emb = self.tdoa_encoder(tdoa_norm)  # (B, N, d_model)

        # 4. Radio features per bird (optional)
        if self.use_radio and "bird_radio_temporal" in batch:
            rt = batch["bird_radio_temporal"]  # (B, N, C, T)
            rt_flat = rt.reshape(B * N_birds, rt.shape[2], rt.shape[3])
            radio_emb = self.radio_encoder(rt_flat).reshape(B, N_birds, -1)
        else:
            radio_emb = None

        # 5. Per-candidate scoring
        audio_expanded = audio_emb.unsqueeze(1).expand(-1, N_birds, -1)
        element_product = audio_expanded * tdoa_emb  # interaction term

        if radio_emb is not None:
            scorer_input = torch.cat([audio_expanded, tdoa_emb, element_product, radio_emb], dim=-1)
        else:
            scorer_input = torch.cat([audio_expanded, tdoa_emb, element_product], dim=-1)

        logits = self.scorer(scorer_input).squeeze(-1)  # (B, N)

        # 6. Mask invalid birds
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

        # 7. Localization from audio
        pred_pos = self.loc_head(audio_emb)

        # 8. Adversarial head (gradient reversed)
        if self.training and self.adv_alpha > 0:
            reversed_emb = GradientReversal.apply(audio_emb, self.adv_alpha)
            adv_logits = self.adv_head(reversed_emb)
        else:
            adv_logits = None

        attn = F.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "pred_position": pred_pos,
            "attn_weights": attn,
            "adv_logits": adv_logits,
        }
