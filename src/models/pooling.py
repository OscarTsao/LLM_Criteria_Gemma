"""Pooling strategies for sentence encoders."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


def _mask_expand(attention_mask: Optional[Tensor], hidden_states: Tensor) -> Tensor:
    if attention_mask is None:
        return torch.ones_like(hidden_states[..., 0], dtype=hidden_states.dtype, device=hidden_states.device)
    return attention_mask.to(dtype=hidden_states.dtype)


class PoolingLayer(nn.Module):
    """Base pooling layer."""

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:  # noqa: D401
        raise NotImplementedError


class MeanPooling(PoolingLayer):
    """Mean pooling with optional attention mask."""

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        mask = _mask_expand(attention_mask, hidden_states).unsqueeze(-1)
        weighted = hidden_states * mask
        summed = weighted.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts


class FirstKPooling(PoolingLayer):
    """Average over the first K tokens."""

    def __init__(self, k: int = 1) -> None:
        super().__init__()
        self.k = k

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        seq_len = hidden_states.size(1)
        k = min(self.k, seq_len)
        indices = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        selector = (indices < k).to(dtype=hidden_states.dtype)
        if attention_mask is not None:
            selector = selector * attention_mask.to(dtype=hidden_states.dtype)
        weighted = hidden_states * selector.unsqueeze(-1)
        summed = weighted.sum(dim=1)
        counts = selector.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
        return summed / counts


class LastKPooling(PoolingLayer):
    """Average over the final K tokens."""

    def __init__(self, k: int = 1) -> None:
        super().__init__()
        self.k = k

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        seq_len = hidden_states.size(1)
        indices = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1, keepdim=True)
        else:
            lengths = torch.full((hidden_states.size(0), 1), seq_len, device=hidden_states.device, dtype=indices.dtype)
        selector = (indices >= (lengths - self.k)).to(dtype=hidden_states.dtype)
        if attention_mask is not None:
            selector = selector * attention_mask.to(dtype=hidden_states.dtype)
        weighted = hidden_states * selector.unsqueeze(-1)
        summed = weighted.sum(dim=1)
        counts = selector.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
        return summed / counts


class AttentionKeyValuePooling(PoolingLayer):
    """Single-query attention with learned key/value projections."""

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
        nn.init.xavier_uniform_(self.query)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        batch_size = hidden_states.size(0)
        query = self.query.expand(batch_size, -1, -1)
        keys = self.key_proj(hidden_states)
        values = self.value_proj(hidden_states)
        scores = torch.matmul(query, keys.transpose(1, 2)) / math.sqrt(keys.size(-1))
        if attention_mask is not None:
            mask = (attention_mask == 0).unsqueeze(1)
            scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        pooled = torch.matmul(weights, values).squeeze(1)
        return pooled


class AttentionQueryPooling(PoolingLayer):
    """Multi-head attention pooling with a learnable query."""

    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
        nn.init.xavier_uniform_(self.query)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        batch_size = hidden_states.size(0)
        query = self.query.expand(batch_size, -1, -1)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        pooled, _ = self.attention(
            query,
            hidden_states,
            hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return pooled.squeeze(1)


def build_pooler(
    strategy: str,
    hidden_size: int,
    first_k: int = 1,
    last_k: int = 1,
    attention_kwargs: Optional[dict] = None,
) -> PoolingLayer:
    strategy = strategy.lower()
    if strategy == "mean":
        return MeanPooling()
    if strategy == "first_k":
        return FirstKPooling(k=first_k)
    if strategy == "last_k":
        return LastKPooling(k=last_k)
    if strategy == "attention_kv":
        kwargs = attention_kwargs or {}
        dropout = kwargs.get("dropout", 0.1)
        return AttentionKeyValuePooling(hidden_size=hidden_size, dropout=dropout)
    if strategy == "attention_query":
        kwargs = attention_kwargs or {}
        num_heads = kwargs.get("num_heads", 4)
        dropout = kwargs.get("dropout", 0.1)
        return AttentionQueryPooling(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
    raise ValueError(f"Unknown pooling strategy '{strategy}'.")
