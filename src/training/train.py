"""Hydra entry point for ReDSM5 sentence-level training."""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from src.data import (
    FoldSplit,
    PostRecord,
    RedSM5DataCollator,
    RedSM5Dataset,
    ensure_folds,
    filter_posts,
    load_redsm5_posts,
    NUM_LABELS,
    LABEL_NAMES,
)
from src.models import ModelBundle, build_model
from src.training.losses import LossConfig, build_loss
from src.training.metrics import MetricResult, compute_metrics

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    tensor_keys = {"input_ids", "attention_mask", "token_type_ids", "labels", "post_labels"}
    result: Dict[str, Any] = {}
    for key, value in batch.items():
        if key in tensor_keys and isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


def _sigmoid_numpy(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def _build_dataloaders(
    train_posts: Sequence[PostRecord],
    val_posts: Sequence[PostRecord],
    tokenizer,
    data_cfg,
    train_cfg,
) -> Tuple[RedSM5Dataset, RedSM5Dataset, DataLoader, DataLoader]:
    level = data_cfg.get("level", "sentence")
    max_len = data_cfg.get("max_len", 512)
    tokenizer_kwargs = data_cfg.get("tokenizer", {})
    train_dataset = RedSM5Dataset.from_posts(train_posts, level=level)
    val_dataset = RedSM5Dataset.from_posts(val_posts, level=level)
    collator = RedSM5DataCollator(
        tokenizer=tokenizer,
        max_length=max_len,
        padding=tokenizer_kwargs.get("padding", True),
        truncation=tokenizer_kwargs.get("truncation", True),
    )
    num_workers = train_cfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def _compute_pos_weight(dataset: RedSM5Dataset) -> torch.Tensor:
    labels_matrix = np.stack([example.labels for example in dataset._examples])  # type: ignore[attr-defined]
    num_samples = labels_matrix.shape[0]
    pos_counts = labels_matrix.sum(axis=0).astype(np.float64)
    pos_counts = np.clip(pos_counts, 1.0, None)
    neg_counts = np.clip(num_samples - pos_counts, 1.0, None)
    pos_weight = torch.tensor(neg_counts / pos_counts, dtype=torch.float32)
    return pos_weight


def _scheduler_factory(
    scheduler_name: str,
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
):
    if scheduler_name == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    if scheduler_name == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    raise ValueError(f"Unknown scheduler: {scheduler_name}")


def _is_improvement(value: float, best: float, mode: str, min_delta: float) -> bool:
    if best is None:
        return True
    if mode == "max":
        return value > best + min_delta
    return value < best - min_delta


@dataclass
class FoldArtifacts:
    fold: int
    metrics: MetricResult
    checkpoint_path: Path
    oof_logits_path: Optional[Path]
    oof_probs_path: Optional[Path]
    oof_labels_path: Optional[Path]
    oof_meta_path: Optional[Path]


def run_fold(
    fold_idx: int,
    posts: Sequence[PostRecord],
    split: FoldSplit,
    cfg: DictConfig,
    device: torch.device,
) -> FoldArtifacts:
    model_bundle: ModelBundle = build_model(cfg.model, num_labels=NUM_LABELS)
    tokenizer = model_bundle.tokenizer

    fold_dir = Path.cwd() / f"fold_{fold_idx}"
    checkpoints_dir = fold_dir / "checkpoints"
    metrics_dir = fold_dir / "metrics"
    predictions_dir = fold_dir / "predictions"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(fold_dir / "tokenizer")

    train_posts = filter_posts(posts, split.train_posts)
    val_posts = filter_posts(posts, split.val_posts)
    train_dataset, val_dataset, train_loader, val_loader = _build_dataloaders(
        train_posts=train_posts,
        val_posts=val_posts,
        tokenizer=tokenizer,
        data_cfg=cfg.data,
        train_cfg=cfg.train,
    )

    model = model_bundle.model.to(device)
    optimizer = AdamW(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=tuple(cfg.train.optim.betas),
        eps=cfg.train.optim.eps,
    )
    num_update_steps_per_epoch = math.ceil(len(train_loader) / cfg.train.grad_accum)
    total_steps = num_update_steps_per_epoch * cfg.train.epochs
    warmup_steps = int(total_steps * cfg.train.warmup_ratio)
    scheduler = _scheduler_factory(cfg.train.scheduler, optimizer, warmup_steps, total_steps)

    amp_enabled = bool(cfg.train.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if cfg.train.amp_precision == "bf16" else torch.float16
    use_scaler = amp_enabled and amp_dtype == torch.float16
    scaler = GradScaler(enabled=use_scaler)

    pos_weight = None
    if cfg.train.loss.class_weights == "inverse_freq":
        pos_weight = _compute_pos_weight(train_dataset).to(device)
    loss_fn = build_loss(
        LossConfig(
            name=cfg.train.loss.name,
            focal_gamma=cfg.train.loss.focal_gamma,
            class_weights=cfg.train.loss.class_weights,
        ),
        pos_weight=pos_weight,
    )

    history: List[Dict[str, Any]] = []
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    best_metrics: Optional[MetricResult] = None
    best_score: Optional[float] = None
    patience_counter = 0
    eval_interval = cfg.train.logging.eval_interval
    best_logits: Optional[np.ndarray] = None
    best_labels_array: Optional[np.ndarray] = None
    best_meta: Optional[List[Dict[str, Any]]] = None
    best_probs: Optional[np.ndarray] = None

    for epoch in range(cfg.train.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = _to_device(batch, device)
            targets = batch["labels"]
            with autocast(enabled=amp_enabled, dtype=amp_dtype if amp_enabled else None):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                )
                loss = loss_fn(outputs.logits, targets)
                loss = loss / cfg.train.grad_accum
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (step + 1) % cfg.train.grad_accum == 0 or (step + 1) == len(train_loader):
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item() * cfg.train.grad_accum
        epoch_loss /= len(train_loader)

        should_eval = ((epoch + 1) % eval_interval == 0) or (epoch + 1 == cfg.train.epochs)
        metrics_map: Dict[str, Any] = {"epoch": epoch + 1, "train_loss": epoch_loss}

        if should_eval:
            model.eval()
            all_logits: List[np.ndarray] = []
            all_labels: List[np.ndarray] = []
            all_meta: List[Dict[str, Any]] = []
            with torch.no_grad():
                for batch in val_loader:
                    batch_device = _to_device(batch, device)
                    outputs = model(
                        input_ids=batch_device["input_ids"],
                        attention_mask=batch_device["attention_mask"],
                        token_type_ids=batch_device.get("token_type_ids"),
                    )
                    logits = outputs.logits.detach().cpu().numpy()
                    labels = batch["labels"].cpu().numpy()
                    all_logits.append(logits)
                    all_labels.append(labels)
                    all_meta.extend(batch["meta"])
            val_logits = np.concatenate(all_logits, axis=0)
            val_labels = np.concatenate(all_labels, axis=0)
            val_probs = _sigmoid_numpy(val_logits)
            metric_result = compute_metrics(
                y_true=val_labels,
                y_score=val_probs,
                threshold=cfg.train.evaluation.threshold,
                label_names=LABEL_NAMES,
                pr_points=cfg.train.evaluation.pr_points,
            )
            metrics_map.update(
                {
                    "val_macro_auprc": metric_result.macro_auprc,
                    "val_macro_f1": metric_result.macro_f1,
                }
            )
            monitor_value = (
                metric_result.macro_auprc
                if cfg.train.early_stopping.monitor == "val/macro_auprc"
                else metric_result.macro_f1
            )
            if best_score is None or _is_improvement(
                monitor_value,
                best_score,
                cfg.train.early_stopping.mode,
                cfg.train.early_stopping.min_delta,
            ):
                best_score = monitor_value
                patience_counter = 0
                best_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                best_metrics = metric_result
                best_logits = val_logits
                best_labels_array = val_labels
                best_probs = val_probs
                best_meta = all_meta
            else:
                patience_counter += 1
                if patience_counter >= cfg.train.early_stopping.patience:
                    LOGGER.info("Early stopping triggered on fold %d at epoch %d.", fold_idx, epoch + 1)
                    history.append(metrics_map)
                    break
        history.append(metrics_map)

    assert best_state_dict is not None and best_metrics is not None, "Training concluded without best state."
    model.load_state_dict(best_state_dict)

    checkpoint_path = checkpoints_dir / cfg.train.checkpoint.filename
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "label_names": LABEL_NAMES,
            "metrics": {
                "macro_auprc": best_metrics.macro_auprc,
                "macro_f1": best_metrics.macro_f1,
                "per_class_f1": best_metrics.per_class_f1,
            },
        },
        checkpoint_path,
    )

    with (metrics_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with (metrics_dir / "best_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "macro_auprc": best_metrics.macro_auprc,
                "macro_f1": best_metrics.macro_f1,
                "per_class_f1": best_metrics.per_class_f1,
                "per_class_pr": best_metrics.per_class_pr,
            },
            handle,
            indent=2,
        )

    oof_logits_path = None
    oof_probs_path = None
    oof_labels_path = None
    oof_meta_path = None
    if cfg.train.logging.save_oof:
        assert (
            best_logits is not None
            and best_probs is not None
            and best_labels_array is not None
            and best_meta is not None
        )
        oof_logits_path = predictions_dir / "oof_logits.npy"
        oof_probs_path = predictions_dir / "oof_probs.npy"
        oof_labels_path = predictions_dir / "oof_labels.npy"
        oof_meta_path = predictions_dir / "ids.json"
        np.save(oof_logits_path, best_logits)
        np.save(oof_probs_path, best_probs)
        np.save(oof_labels_path, best_labels_array)
        with oof_meta_path.open("w", encoding="utf-8") as handle:
            json.dump(best_meta, handle, indent=2)

    return FoldArtifacts(
        fold=fold_idx,
        metrics=best_metrics,
        checkpoint_path=checkpoint_path,
        oof_logits_path=oof_logits_path,
        oof_probs_path=oof_probs_path,
        oof_labels_path=oof_labels_path,
        oof_meta_path=oof_meta_path,
    )


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    data_dir = Path(to_absolute_path(cfg.data.data_dir))

    LOGGER.info("Loading posts from %s", data_dir)
    posts = load_redsm5_posts(data_dir)
    splits_dir = Path(to_absolute_path(cfg.data.splits_dir))
    fold_splits = ensure_folds(posts, splits_dir, folds=cfg.train.folds, seed=cfg.train.seed)

    fold_results: List[FoldArtifacts] = []
    for fold_idx in range(cfg.train.folds):
        LOGGER.info("Starting fold %d/%d", fold_idx + 1, cfg.train.folds)
        artifacts = run_fold(fold_idx, posts, fold_splits[fold_idx], cfg, device)
        fold_results.append(artifacts)
        LOGGER.info(
            "Fold %d macro-AUPRC=%.4f macro-F1=%.4f",
            fold_idx,
            artifacts.metrics.macro_auprc,
            artifacts.metrics.macro_f1,
        )

    summary = {
        "folds": [
            {
                "fold": artifact.fold,
                "macro_auprc": artifact.metrics.macro_auprc,
                "macro_f1": artifact.metrics.macro_f1,
                "checkpoint": str(artifact.checkpoint_path),
                "oof_logits": str(artifact.oof_logits_path) if artifact.oof_logits_path else None,
                "oof_probs": str(artifact.oof_probs_path) if artifact.oof_probs_path else None,
                "oof_labels": str(artifact.oof_labels_path) if artifact.oof_labels_path else None,
                "oof_meta": str(artifact.oof_meta_path) if artifact.oof_meta_path else None,
            }
            for artifact in fold_results
        ]
    }
    summary["mean_macro_auprc"] = float(np.mean([a.metrics.macro_auprc for a in fold_results]))
    summary["mean_macro_f1"] = float(np.mean([a.metrics.macro_f1 for a in fold_results]))
    with (Path.cwd() / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    LOGGER.info("Training finished. Summary written to %s", Path.cwd() / "summary.json")


if __name__ == "__main__":
    main()
