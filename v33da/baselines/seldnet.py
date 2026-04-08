"""SELDnet baseline — DCASE-style Sound Event Localization and Detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.features import MelSpecExtractor


class SELDnet(nn.Module):
    """
    Simplified SELDnet for vocal attribution and localization.

    Input: (B, n_mics, T, n_mels) stacked mel-spectrograms
    CNN pools frequency, GRU processes time, heads predict attribution + position.
    """

    def __init__(
        self,
        n_mics: int = 5,
        sample_rate: int = 24414,
        n_mels: int = 80,
        max_birds: int = 4,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.n_mics = n_mics
        self.max_birds = max_birds
        self.mel = MelSpecExtractor(sample_rate=sample_rate, n_mels=n_mels)

        # CNN: input (B, n_mics, n_mels, T) — pool over frequency (dim 2)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_mics, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((4, 1)),  # pool freq: 80 -> 20
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((4, 1)),  # pool freq: 20 -> 5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((5, 1)),  # pool freq: 5 -> 1
        )
        # After pooling: freq=1, so cnn_out per time step = 256

        # Bi-directional GRU
        self.gru = nn.GRU(
            256, hidden_dim, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.1,
        )

        # SED branch (attribution)
        self.sed_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_birds),
        )

        # DOA branch (3D position, normalized [0,1])
        self.doa_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

    def forward(self, batch: dict) -> dict:
        audio = batch["audio"]  # (B, n_mics, N_samples)
        B = audio.shape[0]

        # Extract mel per mic, stack as channels
        mels = []
        for mi in range(self.n_mics):
            mel = self.mel(audio[:, mi])  # (B, n_mels, T)
            mels.append(mel)
        x = torch.stack(mels, dim=1)  # (B, n_mics, n_mels, T)

        # CNN — pools frequency
        x = self.cnn(x)  # (B, 256, 1, T)
        x = x.squeeze(2).permute(0, 2, 1)  # (B, T, 256)

        # GRU
        x, _ = self.gru(x)  # (B, T, hidden*2)

        # Pool over time
        x = x.mean(dim=1)  # (B, hidden*2)

        # Heads
        logits = self.sed_head(x)  # (B, max_birds)
        position = self.doa_head(x)  # (B, 3)

        # Mask invalid birds — truncate or pad logits to match batch bird count
        if "bird_mask" in batch:
            mask = batch["bird_mask"]  # (B, max_birds_in_batch)
            n_batch_birds = mask.shape[1]
            if n_batch_birds < self.max_birds:
                logits = logits[:, :n_batch_birds]
            elif n_batch_birds > self.max_birds:
                pad = torch.full((B, n_batch_birds - self.max_birds), float("-inf"), device=logits.device)
                logits = torch.cat([logits, pad], dim=1)
            logits = logits.masked_fill(~mask, float("-inf"))
            position = position  # position is always 3D, independent of bird count

        return {
            "logits": logits,
            "pred_position": position,
            "attn_weights": F.softmax(logits, dim=-1),
        }
