"""Whisper Large v3 encoder — per-mic embeddings with VocSim-style pooling."""

import math

import torch
import torch.nn as nn


class WhisperEncoder(nn.Module):
    """
    Frozen Whisper Large v3 encoder with per-mic embedding and
    VocSim-style time-frequency mean pooling.

    Each mic channel is embedded independently through Whisper.
    For each mic: Z = whisper_encoder(mel)  # (T, D)
    Pooling: v = Concat(mean_time(Z), mean_feat(Z))  # (D + T,) -> project to d_out
    All mics concatenated: (n_mics * d_out,) -> project to d_out

    Input: (B, n_mics, N_samples) multi-channel waveform
    Output: (B, 1, d_out) single embedding (or (B, n_mics, d_out) per-mic)
    """

    def __init__(self, d_out: int = 256, model_size: str = "large-v3", n_mics: int = 5):
        super().__init__()
        self.d_out = d_out
        self.n_mics = n_mics
        self.source_sr = 24414
        self.target_sr = 16000

        try:
            import torchaudio
        except ModuleNotFoundError:
            self.resample = None
        else:
            self.resample = torchaudio.transforms.Resample(self.source_sr, self.target_sr)

        import whisper
        self.whisper_model = whisper.load_model(model_size)
        self.whisper_dim = self.whisper_model.dims.n_audio_state  # 1280 for large-v3
        self.whisper_n_mels = getattr(self.whisper_model.dims, "n_mels", 80)
        self.whisper_n_frames = 1500  # Whisper outputs 1500 time frames
        print(f"  Loaded Whisper-{model_size} (dim={self.whisper_dim})")

        # Freeze Whisper
        for p in self.whisper_model.parameters():
            p.requires_grad = False

        # VocSim pooling output dim: mean_time gives (D,), mean_feat gives (T,)
        # v = concat(mean_time, mean_feat) = (D + T,)
        vocsim_dim = self.whisper_dim + self.whisper_n_frames

        # Per-mic projection
        self.mic_proj = nn.Sequential(
            nn.Linear(vocsim_dim, d_out),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Fuse all mics
        self.fuse = nn.Sequential(
            nn.Linear(n_mics * d_out, d_out),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def _resample_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Resample a batch from source_sr to target_sr with a scipy fallback."""
        if self.resample is not None:
            return self.resample(waveform)

        from scipy.signal import resample_poly

        ratio_gcd = math.gcd(self.source_sr, self.target_sr)
        up = self.target_sr // ratio_gcd
        down = self.source_sr // ratio_gcd
        device = waveform.device
        dtype = waveform.dtype
        resampled = []
        for sample in waveform.detach().cpu().numpy():
            y = resample_poly(sample, up, down).astype("float32", copy=False)
            resampled.append(torch.from_numpy(y))
        return torch.nn.utils.rnn.pad_sequence(resampled, batch_first=True).to(device=device, dtype=dtype)

    def _encode_single(self, waveform_16k: torch.Tensor) -> torch.Tensor:
        """Encode a single-channel batch through Whisper.

        Args:
            waveform_16k: (B, N) at 16kHz

        Returns:
            (B, vocsim_dim) VocSim-pooled embedding
        """
        import whisper
        B = waveform_16k.shape[0]
        device = waveform_16k.device

        with torch.no_grad():
            features = []
            for i in range(B):
                audio = whisper.pad_or_trim(waveform_16k[i], 480000)
                mel = whisper.log_mel_spectrogram(audio, n_mels=self.whisper_n_mels).unsqueeze(0)
                mel = mel.to(device)
                enc = self.whisper_model.encoder(mel)  # (1, 1500, D)
                features.append(enc)
            Z = torch.cat(features, dim=0)  # (B, T=1500, D=1280)

        # VocSim pooling: v = Concat(mean_time(Z), mean_feat(Z))
        mean_time = Z.mean(dim=1)  # (B, D) — average over time
        mean_feat = Z.mean(dim=2)  # (B, T) — average over features
        v = torch.cat([mean_time, mean_feat], dim=-1)  # (B, D+T)

        return v

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, N_samples) single-channel — but we need multi-channel.
                      The DETRFusion model passes ref_mic only.
                      We override to use full multi-channel from batch.

        Returns:
            (B, 1, d_out) embedding sequence (single token)
        """
        # Resample
        x = self._resample_waveform(waveform)  # (B, N_16k)

        # Encode single mic through Whisper + VocSim pooling
        v = self._encode_single(x)  # (B, vocsim_dim)
        emb = self.mic_proj(v)  # (B, d_out)

        return emb.unsqueeze(1)  # (B, 1, d_out)

    def forward_multi_mic_tokens(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Full multi-mic forward with per-mic Whisper + VocSim pooling.

        Args:
            audio: (B, n_mics, N_samples) at native sample rate

        Returns:
            (B, n_mics, d_out) per-microphone embeddings
        """
        mic_embeddings = []

        for mi in range(self.n_mics):
            x = self._resample_waveform(audio[:, mi])  # (B, N_16k)
            v = self._encode_single(x)  # (B, vocsim_dim)
            emb = self.mic_proj(v)  # (B, d_out)
            mic_embeddings.append(emb)

        return torch.stack(mic_embeddings, dim=1)  # (B, n_mics, d_out)

    def forward_multi_mic(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Full multi-mic forward with per-mic Whisper + VocSim pooling.

        Args:
            audio: (B, n_mics, N_samples) at native sample rate

        Returns:
            (B, 1, d_out) fused embedding
        """
        per_mic = self.forward_multi_mic_tokens(audio)
        all_mics = per_mic.reshape(per_mic.shape[0], -1)  # (B, n_mics * d_out)
        fused = self.fuse(all_mics)  # (B, d_out)

        return fused.unsqueeze(1)  # (B, 1, d_out)
