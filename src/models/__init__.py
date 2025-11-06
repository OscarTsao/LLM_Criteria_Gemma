"""Gemma encoder models and pooling strategies."""

from .poolers import (
    BasePooler,
    FirstKPooler,
    LastKPooler,
    MeanPooler,
    CLSPooler,
    MaxPooler,
    AttentionPooler,
    get_pooler,
    POOLER_REGISTRY,
)

from .gemma_encoder import (
    GemmaEncoder,
    GemmaClassifier,
    count_parameters,
)

__all__ = [
    # Poolers
    'BasePooler',
    'FirstKPooler',
    'LastKPooler',
    'MeanPooler',
    'CLSPooler',
    'MaxPooler',
    'AttentionPooler',
    'get_pooler',
    'POOLER_REGISTRY',
    # Encoders
    'GemmaEncoder',
    'GemmaClassifier',
    # Utils
    'count_parameters',
]
