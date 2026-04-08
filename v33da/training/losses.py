"""Loss functions for joint vocal attribution + 3D localization."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointLoss(nn.Module):
    """
    Combined loss: λ₁·CE(attribution) + λ₂·L2(localization).

    Attribution: Cross-entropy over bird logits.
    Localization: L2 distance between predicted and true 3D position.
    """

    def __init__(self, lambda_ce: float = 1.0, lambda_l2: float = 0.01, label_smoothing: float = 0.0):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.lambda_l2 = lambda_l2
        self.label_smoothing = label_smoothing

    def forward(self, outputs: dict, batch: dict) -> dict:
        """
        Args:
            outputs: model output dict with 'logits', 'pred_position'
            batch: collated batch with 'label_idx', 'label_position', 'bird_mask'

        Returns:
            dict with 'loss', 'ce_loss', 'l2_loss'
        """
        logits = outputs["logits"]  # (B, N_birds)
        label_idx = batch["label_idx"]  # (B,)

        # Attribution loss: cross-entropy (replace -inf with large negative for numerical stability)
        safe_logits = logits.clamp(min=-1e4)
        ce_loss = F.cross_entropy(safe_logits, label_idx, label_smoothing=self.label_smoothing)

        # Localization loss: L2 on normalized positions
        pred_pos = outputs["pred_position"]  # (B, 3)
        true_pos = batch["label_position"]  # (B, 3)
        l2_loss = F.mse_loss(pred_pos, true_pos)

        loss = self.lambda_ce * ce_loss + self.lambda_l2 * l2_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss.detach(),
            "l2_loss": l2_loss.detach(),
        }
