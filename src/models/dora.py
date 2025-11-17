"""
DoRA (Weight-Decomposed Low-Rank Adaptation) implementation.

Based on "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353)
DoRA decomposes pre-trained weights into magnitude and direction components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class DoRALayer(nn.Module):
    """
    DoRA layer that wraps a linear layer with low-rank adaptation.

    DoRA decomposes weight W into:
        W = m * V
    where m is magnitude and V is direction (normalized).

    The adapted weight becomes:
        W' = m' * (V + B*A) / ||V + B*A||
    where A and B are low-rank matrices.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices (similar to LoRA)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Magnitude vector (learnable)
        self.magnitude = nn.Parameter(torch.ones(out_features))

        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

        # Initialize A with kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, base_weight: torch.Tensor) -> torch.Tensor:
        """
        Compute DoRA-adapted weight.

        Args:
            base_weight: Original frozen weight matrix [out_features, in_features]

        Returns:
            Adapted weight matrix [out_features, in_features]
        """
        # Compute low-rank update: B @ A
        if self.dropout is not None:
            lora_update = self.dropout(self.lora_B @ self.lora_A) * self.scaling
        else:
            lora_update = (self.lora_B @ self.lora_A) * self.scaling

        # Direction: V + ΔV (normalized)
        direction = base_weight + lora_update
        direction_norm = torch.norm(direction, p=2, dim=1, keepdim=True)
        direction = direction / (direction_norm + 1e-8)

        # Apply magnitude: m * (V + ΔV) / ||V + ΔV||
        adapted_weight = self.magnitude.unsqueeze(1) * direction

        return adapted_weight


class LinearWithDoRA(nn.Module):
    """Linear layer with DoRA adaptation."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer

        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Create DoRA layer
        self.dora = DoRALayer(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        # Move DoRA to same device as base layer
        self.dora = self.dora.to(base_layer.weight.device)

        # Initialize magnitude from base weight norms
        with torch.no_grad():
            weight_norms = torch.norm(base_layer.weight, p=2, dim=1)
            self.dora.magnitude.copy_(weight_norms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with DoRA-adapted weights."""
        # Get adapted weight
        adapted_weight = self.dora(self.base_layer.weight)

        # Apply linear transformation with adapted weight
        output = F.linear(x, adapted_weight, self.base_layer.bias)

        return output


def apply_dora_to_model(
    model: nn.Module,
    target_modules: list = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Apply DoRA to specific modules in a model.

    Args:
        model: The model to adapt
        target_modules: List of module name patterns to apply DoRA to
                       (e.g., ['q_proj', 'v_proj'] for attention query/value)
        rank: Rank for low-rank decomposition
        alpha: Scaling factor
        dropout: Dropout probability

    Returns:
        Model with DoRA layers applied
    """
    if target_modules is None:
        # Default: apply to attention query and value projections
        target_modules = ['q_proj', 'v_proj']

    # Count replacements
    num_replaced = 0

    # Recursively replace linear layers
    for name, module in model.named_modules():
        # Check if this module should be adapted
        should_adapt = any(target in name for target in target_modules)

        if should_adapt and isinstance(module, nn.Linear):
            # Get parent module and attribute name
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            # Create DoRA-adapted layer
            dora_layer = LinearWithDoRA(
                base_layer=module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )

            # Replace the module
            setattr(parent, attr_name, dora_layer)
            num_replaced += 1

    print(f"Applied DoRA to {num_replaced} layers")
    return model


def count_trainable_parameters(model: nn.Module) -> tuple:
    """
    Count trainable and total parameters.

    Returns:
        (trainable_params, total_params, trainable_percentage)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    percentage = 100 * trainable / total if total > 0 else 0

    return trainable, total, percentage
