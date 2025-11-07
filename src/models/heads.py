"""Classifier heads for multi-label sentence models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


class MultiLabelClassificationHead(nn.Module):
    """Feed-forward head for multi-label classification."""

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1,
        hidden_dim: Optional[int] = None,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        layers = [nn.Dropout(dropout)]
        if hidden_dim and hidden_dim > 0:
            layers.append(nn.Linear(hidden_size, hidden_dim))
            if activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            else:
                raise ValueError(f"Unsupported activation '{activation}'.")
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, num_labels))
        else:
            layers.append(nn.Linear(hidden_size, num_labels))
        self.network = nn.Sequential(*layers)

    def forward(self, pooled_embeddings: Tensor) -> Tensor:
        return self.network(pooled_embeddings)


class TokenRationaleHead(nn.Module):
    """Optional token-level rationale head."""

    def __init__(
        self,
        hidden_size: int,
        num_labels: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states: Tensor) -> Tensor:
        dropped = self.dropout(hidden_states)
        return self.classifier(dropped)


@dataclass
class ModelOutput:
    """Container for model outputs."""

    logits: Tensor
    pooled: Tensor
    hidden_states: Tensor
    rationale_logits: Optional[Tensor] = None
