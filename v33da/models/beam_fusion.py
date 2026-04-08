"""BeamFusion — learned beamforming + transformer for vocal attribution."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .encoders.melspec import MelSpecEncoder
from .spatial.conformer import SpatialConformer
from .decoder import FusionDecoder
from ..data.features import SALSALiteExtractor
from ..data.dataset import MIC_POSITIONS, CAGE_X, CAGE_Y, CAGE_Z


# Speed of sound in mm/s
SPEED_OF_SOUND = 343000.0


class LearnedBeamformer(nn.Module):
    """
    Per-bird learned spatial filter based on steering vectors.

    For each bird position, computes physics-based time delays to each mic,
    then applies a learnable filter-and-sum to the multi-channel STFT.
    Output: per-bird "beamformed" spectrogram focused on that bird's location.
    """

    def __init__(self, n_mics: int = 5, n_fft: int = 512, d_model: int = 256):
        super().__init__()
        self.n_mics = n_mics
        self.n_fft = n_fft
        self.n_freq = n_fft // 2 + 1

        # Mic positions as buffer (not parameter)
        self.register_buffer('mic_pos', torch.tensor(MIC_POSITIONS, dtype=torch.float32))

        # Cage bounds for denormalization
        bounds = torch.tensor([CAGE_X, CAGE_Y, CAGE_Z], dtype=torch.float32)
        self.register_buffer('cage_lo', bounds[:, 0])
        self.register_buffer('cage_hi', bounds[:, 1])

        # Learnable refinement of beamformer weights per frequency bin
        # Physics gives the steering vector, this learns corrections
        self.weight_net = nn.Sequential(
            nn.Linear(n_mics * 2, 64),  # real + imag steering components
            nn.ReLU(),
            nn.Linear(64, n_mics * 2),  # refined complex weights
        )

        # Encode beamformed spectrogram to embedding
        self.spec_encoder = nn.Sequential(
            nn.Conv1d(self.n_freq, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def _denorm_position(self, pos_norm):
        """(B, N, 3) normalized -> mm"""
        return pos_norm * (self.cage_hi - self.cage_lo) + self.cage_lo

    def _compute_steering_vector(self, bird_pos_mm, sample_rate=24414.0):
        """
        Compute steering vectors for given bird positions.

        Args:
            bird_pos_mm: (B, N_birds, 3) in mm

        Returns:
            steering: (B, N_birds, n_freq, n_mics) complex steering vectors
        """
        B, N, _ = bird_pos_mm.shape

        # Distance from each bird to each mic: (B, N, n_mics)
        # mic_pos: (n_mics, 3)
        dists = torch.cdist(bird_pos_mm, self.mic_pos.unsqueeze(0).expand(B, -1, -1))

        # Time delay relative to first mic: (B, N, n_mics)
        delays = (dists - dists[:, :, :1]) / SPEED_OF_SOUND  # seconds

        # Frequency bins
        freqs = torch.linspace(0, sample_rate / 2, self.n_freq, device=bird_pos_mm.device)

        # Phase shifts: (B, N, n_freq, n_mics)
        # phase = -2π * f * τ
        phase = -2 * np.pi * freqs[None, None, :, None] * delays[:, :, None, :]

        # Complex steering vector
        steering = torch.complex(torch.cos(phase), torch.sin(phase))

        return steering

    def forward(self, stft_complex, bird_positions_norm, sample_rate=24414.0):
        """
        Apply learned beamforming for each bird.

        Args:
            stft_complex: (B, n_mics, n_freq, T) complex STFT
            bird_positions_norm: (B, N_birds, 3) normalized [0,1]

        Returns:
            per_bird_emb: (B, N_birds, d_model) beamformed embeddings
        """
        B, n_mics, n_freq, T = stft_complex.shape
        N_birds = bird_positions_norm.shape[1]

        # Denormalize positions
        bird_pos_mm = self._denorm_position(bird_positions_norm)

        # Physics-based steering vectors
        steering = self._compute_steering_vector(bird_pos_mm, sample_rate)
        # (B, N_birds, n_freq, n_mics)

        # Learnable weight refinement
        # Concat real and imag parts of steering as features
        steer_features = torch.cat([steering.real, steering.imag], dim=-1)
        # (B, N_birds, n_freq, n_mics*2)
        refined = self.weight_net(steer_features)
        # (B, N_birds, n_freq, n_mics*2)
        weights = torch.complex(refined[..., :n_mics], refined[..., n_mics:])
        # (B, N_birds, n_freq, n_mics)

        # Apply beamformer: weighted sum across mics
        # stft: (B, n_mics, n_freq, T) -> (B, 1, n_mics, n_freq, T)
        # weights: (B, N_birds, n_freq, n_mics) -> (B, N_birds, n_mics, n_freq, 1)
        stft_expanded = stft_complex.unsqueeze(1)  # (B, 1, n_mics, n_freq, T)
        weights_expanded = weights.permute(0, 1, 3, 2).unsqueeze(-1)  # (B, N, n_mics, n_freq, 1)

        # Beamformed signal: sum over mics
        beamformed = (stft_expanded * weights_expanded).sum(dim=2)  # (B, N_birds, n_freq, T)

        # Take magnitude
        beamformed_mag = beamformed.abs()  # (B, N_birds, n_freq, T)

        # Encode each bird's beamformed spectrogram
        # Reshape to process all birds as batch
        bf_flat = beamformed_mag.reshape(B * N_birds, n_freq, T)
        emb_flat = self.spec_encoder(bf_flat).squeeze(-1)  # (B*N, d_model)
        per_bird_emb = emb_flat.reshape(B, N_birds, -1)  # (B, N_birds, d_model)

        return per_bird_emb


class BeamFusion(nn.Module):
    """
    Learned beamforming + transformer fusion model.

    Architecture:
        1. STFT of multi-channel audio
        2. Per-bird learned beamforming → per-bird audio embedding (queries)
        3. Global spatial stream (SALSA + Conformer) → context (keys/values)
        4. Optional: radio features concatenated to bird queries
        5. Transformer decoder cross-attention
        6. Attribution + localization heads
    """

    def __init__(
        self,
        modality: str = "audio_3d_radio",
        d_model: int = 256,
        n_decoder_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        sample_rate: int = 24414,
        n_mels: int = 80,
        n_mics: int = 5,
        n_fft: int = 512,
        hop_length: int = 256,
    ):
        super().__init__()
        self.n_mics = n_mics
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.d_model = d_model
        self.modality = modality

        # Learned beamformer
        self.beamformer = LearnedBeamformer(
            n_mics=n_mics, n_fft=n_fft, d_model=d_model,
        )

        # Radio feature encoder (optional)
        if modality == "audio_3d_radio":
            self.radio_encoder = nn.Sequential(
                nn.BatchNorm1d(12),
                nn.Linear(12, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.query_fuse = nn.Linear(d_model * 2, d_model)
        else:
            self.radio_encoder = None

        # Global spatial context stream
        self.salsa = SALSALiteExtractor(
            n_mics=n_mics, sample_rate=sample_rate, n_mels=n_mels, ref_mic=0,
        )
        self.spatial_encoder = SpatialConformer(
            n_input_channels=n_mics, n_mels=n_mels, d_model=d_model,
            n_layers=2, n_heads=n_heads, dropout=dropout,
        )

        # Fusion decoder
        self.decoder = FusionDecoder(
            d_model=d_model, n_heads=n_heads, n_layers=n_decoder_layers,
            dim_feedforward=d_model * 4, dropout=dropout,
        )

        # Task heads
        self.attribution_head = nn.Linear(d_model, 1)

    def forward(self, batch: dict) -> dict:
        audio = batch["audio"]  # (B, n_mics, N)
        B = audio.shape[0]

        # Compute STFT
        window = torch.hann_window(self.n_fft, device=audio.device)
        stft_list = []
        for mi in range(self.n_mics):
            s = torch.stft(audio[:, mi], self.n_fft, self.hop_length,
                          window=window, return_complex=True)  # (B, n_freq, T)
            stft_list.append(s)
        stft_complex = torch.stack(stft_list, dim=1)  # (B, n_mics, n_freq, T)

        # Per-bird beamformed queries
        queries = self.beamformer(
            stft_complex, batch["bird_positions"], self.sample_rate
        )  # (B, N_birds, d_model)

        # Add radio features if available
        if self.radio_encoder is not None:
            radio = batch["bird_radio"]  # (B, N_birds, 12)
            N_birds = radio.shape[1]
            radio_flat = radio.reshape(B * N_birds, -1)
            radio_emb = self.radio_encoder(radio_flat).reshape(B, N_birds, -1)
            queries = self.query_fuse(torch.cat([queries, radio_emb], dim=-1))

        # Global spatial context
        salsa_features = self.salsa(audio)  # (B, C, T, F)
        context = self.spatial_encoder(salsa_features)  # (B, T_sp, D)

        # Decode
        decoded = self.decoder(queries, context, batch["bird_mask"])  # (B, N_birds, D)

        # Attribution
        logits = self.attribution_head(decoded).squeeze(-1)  # (B, N_birds)
        logits = logits.masked_fill(~batch["bird_mask"], float("-inf"))

        # Localization (soft-argmax)
        attn_weights = F.softmax(logits, dim=-1)
        pred_position = (attn_weights.unsqueeze(-1) * batch["bird_positions"]).sum(dim=1)

        return {
            "logits": logits,
            "pred_position": pred_position,
            "attn_weights": attn_weights,
        }
