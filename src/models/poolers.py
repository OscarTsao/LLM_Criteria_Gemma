"""Pooling strategies for sequence models."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional


class BasePooler(ABC):
    """Abstract base class for pooling strategies."""

    @abstractmethod
    def __call__(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass


class FirstKPooler(BasePooler):
    """Takes the first K tokens."""
    def __init__(self, k: int = 1):
        self.k = k

    def __call__(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return hidden_states[:, :self.k, :].mean(dim=1)


class LastKPooler(BasePooler):
    """Takes the last K tokens."""
    def __init__(self, k: int = 1):
        self.k = k

    def __call__(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return hidden_states[:, -self.k:, :].mean(dim=1)


class MeanPooler(BasePooler):
    """Average pooling over valid tokens."""
    def __call__(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.shape)
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1)
        return sum_hidden / sum_mask.clamp(min=1e-9)


class CLSPooler(BasePooler):
    """Takes the [CLS] token."""
    def __call__(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return hidden_states[:, 0, :]


class MaxPooler(BasePooler):
    """Max pooling over valid tokens."""
    def __call__(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.max(dim=1)[0]
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.shape)
        hidden_states = hidden_states.clone()
        hidden_states[~mask_expanded.bool()] = float('-inf')
        return hidden_states.max(dim=1)[0]


class AttentionPooler(nn.Module):
    """Learned attention pooling."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.query)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = hidden_states.size(0)
        query = self.query.expand(batch_size, -1, -1)
        keys = self.key_proj(hidden_states)
        values = self.value_proj(hidden_states)
        scores = torch.matmul(query, keys.transpose(1, 2)) / (keys.size(-1) ** 0.5)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, values).squeeze(1)


POOLER_REGISTRY = {
    'first_k': FirstKPooler,
    'last_k': LastKPooler,
    'mean': MeanPooler,
    'cls': CLSPooler,
    'max': MaxPooler,
    'attention': AttentionPooler,
}


def get_pooler(pooler_name: str, hidden_dim: Optional[int] = None, **kwargs):
    """Get a pooler instance by name."""
    if pooler_name not in POOLER_REGISTRY:
        raise ValueError(f"Unknown pooler: {pooler_name}")
    pooler_class = POOLER_REGISTRY[pooler_name]
    if pooler_name == 'attention':
        if hidden_dim is None:
            raise ValueError("AttentionPooler requires 'hidden_dim'")
        return pooler_class(hidden_dim, **kwargs)
    return pooler_class(**kwargs)
