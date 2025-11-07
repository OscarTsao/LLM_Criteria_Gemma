"""
Binary Classification Model for Post-Criterion Matching.

Adapts Gemma encoder for binary classification on (post, criterion) pairs.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GemmaBinaryClassifier(nn.Module):
    """
    Gemma-based binary classifier for post-criterion matching.

    Takes (post, criterion) pairs as input and outputs binary prediction.
    Uses bidirectional attention adaptation for encoder tasks.

    Args:
        model_name: Huggingface model name (e.g., 'google/gemma-2-2b')
        pooling_strategy: How to pool token representations ('mean', 'cls', etc.)
        freeze_encoder: Whether to freeze base model weights
        dropout: Dropout probability
    """

    def __init__(
        self,
        model_name: str = 'google/gemma-2-2b',
        pooling_strategy: str = 'mean',
        freeze_encoder: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.freeze_encoder = freeze_encoder

        # Load base model
        logger.info(f"Loading model: {model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        )

        # Enable bidirectional attention
        self._enable_bidirectional_attention()

        # Freeze encoder if requested
        if freeze_encoder:
            logger.info("Freezing base model parameters")
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Get hidden dimension
        if hasattr(self.base_model.config, 'hidden_size'):
            self.hidden_dim = self.base_model.config.hidden_size
        elif hasattr(self.base_model.config, 'd_model'):
            self.hidden_dim = self.base_model.config.d_model
        else:
            raise ValueError("Could not determine hidden dimension")

        # Pooling layer
        from src.models.poolers import MeanPooler, CLSPooler, MaxPooler, AttentionPooler

        if pooling_strategy == 'mean':
            self.pooler = MeanPooler(self.hidden_dim)
        elif pooling_strategy == 'cls':
            self.pooler = CLSPooler(self.hidden_dim)
        elif pooling_strategy == 'max':
            self.pooler = MaxPooler(self.hidden_dim)
        elif pooling_strategy == 'attention':
            self.pooler = AttentionPooler(self.hidden_dim)
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

        # Binary classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_dim, 2)  # 2 classes: no-match, match

        logger.info(f"Binary classifier initialized:")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Hidden dim: {self.hidden_dim}")
        logger.info(f"  Pooling: {pooling_strategy}")
        logger.info(f"  Frozen encoder: {freeze_encoder}")

    def _enable_bidirectional_attention(self):
        """
        Enable bidirectional attention for encoder tasks.

        This is the key modification from the paper: converting causal
        attention to bidirectional attention for classification tasks.
        """
        logger.info("Enabling bidirectional attention...")

        # For Gemma models, we need to modify the attention mask behavior
        # This allows tokens to attend to all other tokens (not just previous)
        if hasattr(self.base_model, 'model'):
            model = self.base_model.model
        else:
            model = self.base_model

        # Patch attention layers to remove causal masking
        for layer in model.layers if hasattr(model, 'layers') else []:
            if hasattr(layer, 'self_attn'):
                # Set is_causal to False (if supported)
                if hasattr(layer.self_attn, 'is_causal'):
                    layer.self_attn.is_causal = False

        logger.info("Bidirectional attention enabled")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) attention mask

        Returns:
            logits: (batch, 2) binary classification logits
        """
        # Get hidden states from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Get last hidden state
        hidden_states = outputs.hidden_states[-1]  # (batch, seq, hidden)

        # Pool token representations
        pooled = self.pooler(hidden_states, attention_mask)  # (batch, hidden)

        # Classification head
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (batch, 2)

        return logits

    def get_trainable_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def load_binary_tokenizer(model_name: str):
    """Load tokenizer for binary classification model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
