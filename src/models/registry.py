"""Model registry and factory functions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .heads import ModelOutput, MultiLabelClassificationHead, TokenRationaleHead
from .pooling import PoolingLayer, build_pooler

try:  # pragma: no cover - optional dependency
    from omegaconf import DictConfig, OmegaConf
except ImportError:  # pragma: no cover
    DictConfig = None  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:  # pragma: no cover - optional dependency
    LoraConfig = None  # type: ignore[assignment]
    TaskType = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    """Container for the instantiated model and tokenizer."""

    model: nn.Module
    tokenizer: PreTrainedTokenizerBase
    hidden_size: int


class SentenceClassificationModel(nn.Module):
    """End-to-end encoder + pooling + classifier."""

    def __init__(
        self,
        encoder: nn.Module,
        pooler: PoolingLayer,
        classifier: MultiLabelClassificationHead,
        rationale_head: Optional[TokenRationaleHead] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooler = pooler
        self.classifier = classifier
        self.rationale_head = rationale_head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> ModelOutput:
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "output_hidden_states": True,
            "use_cache": False,
            "return_dict": True,
        }
        encoder_kwargs.update(kwargs)
        outputs = self.encoder(**encoder_kwargs)
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None:
            hidden_states = outputs.hidden_states[-1]
        pooled = self.pooler(hidden_states, attention_mask)
        logits = self.classifier(pooled)
        rationale_logits = None
        if self.rationale_head is not None:
            rationale_logits = self.rationale_head(hidden_states)
        return ModelOutput(
            logits=logits,
            pooled=pooled,
            hidden_states=hidden_states,
            rationale_logits=rationale_logits,
        )

    def resize_token_embeddings(self, new_vocab_size: int) -> None:
        if hasattr(self.encoder, "resize_token_embeddings"):
            self.encoder.resize_token_embeddings(new_vocab_size)


def _to_dict(config: Union[Mapping[str, Any], Any]) -> Dict[str, Any]:
    if DictConfig is not None and isinstance(config, DictConfig):  # type: ignore[arg-type]
        assert OmegaConf is not None
        return dict(OmegaConf.to_container(config, resolve=True))  # type: ignore[union-attr]
    if isinstance(config, Mapping):
        return dict(config)
    return dict(config.__dict__)


def _ensure_pad_token(tokenizer: PreTrainedTokenizerBase) -> None:
    if tokenizer.pad_token_id is not None:
        return
    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return
    tokenizer.add_special_tokens({"pad_token": "<pad>"})


def _resolve_attention_kwargs(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    attention_cfg = model_cfg.get("attention_pool") or {}
    return attention_cfg


def _resolve_lora_targets(model_name: str, override: Optional[Sequence[str]]) -> Sequence[str]:
    if override:
        return list(override)
    name = model_name.lower()
    if "deberta" in name:
        return ["query_proj", "value_proj"]
    if "roberta" in name or "bert" in name:
        return ["query", "value"]
    if "modernbert" in name:
        return ["query", "value"]
    if "gemma" in name or "llama" in name:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    return ["query", "value"]


def _apply_lora_if_needed(model: nn.Module, model_cfg: Dict[str, Any], model_name: str) -> nn.Module:
    peft_cfg = model_cfg.get("peft") or {}
    if not peft_cfg.get("enable_lora", False):
        return model
    if get_peft_model is None:
        raise ImportError("peft is required for LoRA but is not installed.")
    target_modules = _resolve_lora_targets(model_name, peft_cfg.get("target_modules"))
    lora_config = LoraConfig(
        r=peft_cfg.get("lora_r", 16),
        lora_alpha=peft_cfg.get("lora_alpha", 32),
        lora_dropout=peft_cfg.get("lora_dropout", 0.05),
        bias="none",
        target_modules=target_modules,
        task_type=TaskType.SEQ_CLS,
    )
    LOGGER.info("Applying LoRA to modules: %s", target_modules)
    return get_peft_model(model, lora_config)


def _freeze_encoder_if_requested(encoder: nn.Module, freeze: bool, lora_enabled: bool) -> None:
    if not freeze:
        return
    if lora_enabled:
        LOGGER.warning("freeze_base_model=True with LoRA enabled. LoRA adapters remain trainable.")
    for param in encoder.parameters():
        param.requires_grad = False


def _enable_gradient_checkpointing(encoder: nn.Module) -> None:
    if hasattr(encoder, "gradient_checkpointing_enable"):
        encoder.gradient_checkpointing_enable()
    if hasattr(encoder, "enable_input_require_grads"):
        encoder.enable_input_require_grads()


def _load_encoder(model_cfg: Dict[str, Any]) -> Tuple[nn.Module, PretrainedConfig, str]:
    model_name = model_cfg["model_name"]
    torch_dtype_str = model_cfg.get("torch_dtype", None)
    torch_dtype = getattr(torch, torch_dtype_str) if isinstance(torch_dtype_str, str) else None
    load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    name_lower = model_name.lower()
    if "gemma" in name_lower:
        base_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        base_model.config.is_decoder = False
        base_model.config.use_cache = False
        if hasattr(base_model, "model"):
            encoder = base_model.model
        else:
            encoder = base_model
        for layer in getattr(encoder, "layers", []):
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                if hasattr(attn, "is_causal"):
                    attn.is_causal = False
                if hasattr(attn, "causal"):
                    attn.causal = False
        return encoder, encoder.config, model_name
    encoder = AutoModel.from_pretrained(model_name, **load_kwargs)
    encoder.config = config
    return encoder, encoder.config, model_name


def build_model(
    model_cfg: Union[Mapping[str, Any], Any],
    num_labels: int,
) -> ModelBundle:
    cfg = _to_dict(model_cfg)
    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    _ensure_pad_token(tokenizer)
    tokenizer.padding_side = "right"

    encoder, encoder_config, resolved_name = _load_encoder(cfg)
    hidden_size = getattr(encoder_config, "hidden_size", None)
    if hidden_size is None:
        raise AttributeError(f"Encoder config for {resolved_name} does not expose hidden_size.")

    dropout = cfg.get("hidden_dropout", 0.1)
    classifier_hidden_dim = cfg.get("classifier_hidden_dim")
    pooling_strategy = cfg.get("pooling_strategy", "mean")
    first_k = cfg.get("first_k", 1)
    last_k = cfg.get("last_k", 1)
    attention_kwargs = _resolve_attention_kwargs(cfg)

    pooler = build_pooler(
        strategy=pooling_strategy,
        hidden_size=hidden_size,
        first_k=first_k,
        last_k=last_k,
        attention_kwargs=attention_kwargs,
    )
    classifier = MultiLabelClassificationHead(
        hidden_size=hidden_size,
        num_labels=num_labels,
        dropout=dropout,
        hidden_dim=classifier_hidden_dim,
    )

    rationale_cfg = cfg.get("rationale_head") or {}
    rationale_head: Optional[TokenRationaleHead] = None
    if rationale_cfg.get("enabled", False):
        rationale_head = TokenRationaleHead(
            hidden_size=hidden_size,
            num_labels=rationale_cfg.get("num_labels", 1),
            dropout=rationale_cfg.get("dropout", dropout),
        )

    freeze = cfg.get("freeze_base_model", False)
    lora_enabled = cfg.get("peft", {}).get("enable_lora", False)
    _freeze_encoder_if_requested(
        encoder=encoder,
        freeze=freeze,
        lora_enabled=lora_enabled,
    )
    encoder = _apply_lora_if_needed(encoder, cfg, model_name)
    if cfg.get("grad_checkpointing", False):
        _enable_gradient_checkpointing(encoder)

    model = SentenceClassificationModel(
        encoder=encoder,
        pooler=pooler,
        classifier=classifier,
        rationale_head=rationale_head,
    )

    if hasattr(model, "resize_token_embeddings") and len(tokenizer) > getattr(encoder_config, "vocab_size", 0):
        model.resize_token_embeddings(len(tokenizer))

    return ModelBundle(model=model, tokenizer=tokenizer, hidden_size=hidden_size)
