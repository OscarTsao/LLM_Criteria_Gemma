"""Gemma Encoder and Classifier for sequence classification."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Union, Tuple
from .poolers import MeanPooler, CLSPooler, MaxPooler, AttentionPooler


class GemmaEncoder(nn.Module):
    """Bidirectional Gemma encoder with pooling."""

    def __init__(
        self,
        model_name: str = "google/gemma-2b",
        pooling_strategy: str = "mean",
        freeze_encoder: bool = False,
        device: Optional[str] = None,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.config.use_cache = False

        # Enable gradient checkpointing to reduce memory usage
        if use_gradient_checkpointing:
            print("Enabling gradient checkpointing for memory efficiency...")
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, 'enable_input_require_grads'):
                self.model.enable_input_require_grads()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        hidden_size = self.model.config.hidden_size

        # Initialize pooler
        if pooling_strategy == "mean":
            self.pooler = MeanPooler()
        elif pooling_strategy == "cls":
            self.pooler = CLSPooler()
        elif pooling_strategy == "max":
            self.pooler = MaxPooler()
        elif pooling_strategy == "attention":
            self.pooler = AttentionPooler(hidden_size)
        else:
            raise ValueError(f"Unknown pooling: {pooling_strategy}")

        # Move pooler to device if it's an nn.Module
        if isinstance(self.pooler, nn.Module):
            self.pooler = self.pooler.to(self.device)

        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through encoder."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]

        if isinstance(self.pooler, AttentionPooler):
            return self.pooler(hidden_states, attention_mask)
        return self.pooler(hidden_states, attention_mask)


class GemmaClassifier(nn.Module):
    """Classification model with GemmaEncoder + classifier head."""

    def __init__(
        self,
        num_classes: int,
        model_name: str = "google/gemma-2b",
        pooling_strategy: str = "mean",
        freeze_encoder: bool = False,
        hidden_dropout_prob: float = 0.1,
        classifier_hidden_size: Optional[int] = None,
        device: Optional[str] = None,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = GemmaEncoder(
            model_name=model_name,
            pooling_strategy=pooling_strategy,
            freeze_encoder=freeze_encoder,
            device=device,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        hidden_size = self.encoder.model.config.hidden_size

        # Build classifier
        if classifier_hidden_size:
            self.classifier = nn.Sequential(
                nn.Dropout(hidden_dropout_prob),
                nn.Linear(hidden_size, classifier_hidden_size),
                nn.GELU(),
                nn.Dropout(hidden_dropout_prob),
                nn.Linear(classifier_hidden_size, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(hidden_dropout_prob),
                nn.Linear(hidden_size, num_classes),
            )

        # Move classifier to device
        self.classifier = self.classifier.to(self.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass."""
        pooled = self.encoder(input_ids, attention_mask)
        logits = self.classifier(pooled)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return loss, logits
        return logits


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
