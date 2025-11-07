"""Loss functions for multi-label training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class LossConfig:
    """Configuration for building loss functions."""

    name: str = "bce"
    focal_gamma: float = 1.5
    class_weights: Optional[str] = None


class FocalLoss(nn.Module):
    """Binary focal loss for multi-label classification."""

    def __init__(
        self,
        gamma: float = 1.5,
        pos_weight: Optional[Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("pos_weight", pos_weight, persistent=False)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal = (1.0 - pt).pow(self.gamma) * bce
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def build_loss(
    cfg: LossConfig,
    pos_weight: Optional[Tensor] = None,
) -> nn.Module:
    """Instantiate loss function from config."""

    if cfg.name.lower() == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if cfg.name.lower() == "focal":
        return FocalLoss(gamma=cfg.focal_gamma, pos_weight=pos_weight)
    raise ValueError(f"Unknown loss: {cfg.name}")
