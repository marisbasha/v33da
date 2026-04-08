"""Transformer decoder: candidate queries cross-attend to audio context."""

import torch
import torch.nn as nn


class FusionDecoder(nn.Module):
    """
    Transformer decoder layers where candidate bird queries (Q)
    cross-attend to concatenated audio context (K, V).

    Uses standard PyTorch TransformerDecoderLayer.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4, n_layers: int = 2,
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)

    def forward(self, queries: torch.Tensor, context: torch.Tensor,
                query_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            queries: (B, N_birds, d_model) — candidate query embeddings
            context: (B, T_ctx, d_model) — audio context (identity + spatial)
            query_mask: (B, N_birds) bool — True for valid birds

        Returns:
            (B, N_birds, d_model) — decoded representations
        """
        # TransformerDecoder expects key_padding_mask as True for PADDED positions
        tgt_key_padding_mask = None
        if query_mask is not None:
            tgt_key_padding_mask = ~query_mask  # invert: True = ignore

        out = self.decoder(
            tgt=queries,
            memory=context,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return out
