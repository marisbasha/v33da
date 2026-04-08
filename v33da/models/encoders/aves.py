"""AVES encoder — frozen HuBERT pretrained on animal audio."""

import torch
import torch.nn as nn


class AVESEncoder(nn.Module):
    """
    AVES (Animal Vocalization Encoder based on Self-supervision).
    Uses torchaudio's HuBERT with AVES weights.

    Input: (B, N_samples) single-channel waveform at 16kHz
    Output: (B, T, d_out) embedding sequence
    """

    def __init__(self, d_out: int = 256, model_name: str = "aves-base"):
        super().__init__()
        self.d_out = d_out
        self.target_sr = 16000
        self.source_sr = 24414

        # Load AVES via torchaudio's HuBERT
        import torchaudio
        from torchaudio.models import wav2vec2_model

        # AVES uses HuBERT-base architecture
        config = {
            "extractor_mode": "group_norm",
            "extractor_conv_layer_config": [
                (512, 10, 5), (512, 3, 2), (512, 3, 2),
                (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2),
            ],
            "extractor_conv_bias": False,
            "encoder_embed_dim": 768,
            "encoder_projection_dropout": 0.0,
            "encoder_pos_conv_kernel": 128,
            "encoder_pos_conv_groups": 16,
            "encoder_num_layers": 12,
            "encoder_num_heads": 12,
            "encoder_attention_dropout": 0.0,
            "encoder_ff_interm_features": 3072,
            "encoder_ff_interm_dropout": 0.0,
            "encoder_dropout": 0.0,
            "encoder_layer_norm_first": False,
            "encoder_layer_drop": 0.0,
            "aux_num_out": None,
        }
        self.encoder = wav2vec2_model(**config)

        # Try to load AVES weights
        try:
            aves_path = self._download_aves()
            state = torch.load(aves_path, map_location="cpu", weights_only=True)
            self.encoder.load_state_dict(state, strict=False)
            print(f"  Loaded AVES weights from {aves_path}")
        except Exception as e:
            print(f"  Warning: Could not load AVES weights ({e}), using random init")

        # Freeze
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Resample
        self.resample = torchaudio.transforms.Resample(self.source_sr, self.target_sr)

        # Project to d_out
        self.proj = nn.Linear(768, d_out)

    def _download_aves(self):
        """Download AVES checkpoint if not cached."""
        from pathlib import Path
        import urllib.request
        cache_dir = Path.home() / ".cache" / "aves"
        cache_dir.mkdir(parents=True, exist_ok=True)
        aves_path = cache_dir / "aves-base-bio.pt"
        if not aves_path.exists():
            # AVES weights hosted on GitHub releases by Earth Species Project
            url = "https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-bio.torchaudio.pt"
            print(f"  Downloading AVES weights from {url}...")
            urllib.request.urlretrieve(url, aves_path)
        return aves_path

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """(B, N_samples) -> (B, T, d_out)"""
        # Resample to 16kHz
        x = self.resample(waveform)

        # AVES/HuBERT expects (B, N)
        with torch.no_grad():
            features, _ = self.encoder.extract_features(x)
            # features is a list of layer outputs, take the last
            x = features[-1]  # (B, T, 768)

        x = self.proj(x)  # (B, T, d_out)
        return x
