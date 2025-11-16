"""
5-Fold Cross-Validation Training for Binary NLI Criteria Matching

Trains Gemma Encoder on NLI-style text-pair classification:
- Input: [CLS] post [SEP] criterion [SEP]
- Output: Binary (matched/unmatched)
- 5-fold stratified CV with Hydra configuration
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_auc_score
)
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import json
from tqdm.auto import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.gemma_encoder import GemmaClassifier
from data.nli_cv_splits import create_nli_cv_splits, load_nli_fold_split, get_nli_fold_metadata
from data.redsm5_nli_dataset import get_class_weights, NUM_CLASSES

logger = logging.getLogger(__name__)


class NLIFoldTrainer:
    """Trainer for single fold of NLI binary classification."""

    def __init__(
        self,
        model: GemmaClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: DictConfig,
        fold_num: int,
        output_dir: Path,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.fold_num = fold_num
        self.output_dir = output_dir
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # Scheduler
        total_steps = len(train_loader) * config.training.num_epochs
        warmup_steps = int(total_steps * config.training.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Loss function (binary classification with class weights)
        if config.training.use_class_weights:
            class_weights = get_class_weights(train_loader.dataset).to(device)
            logger.info(f"Using class weights: {class_weights.cpu().numpy()}")
        else:
            class_weights = None

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Mixed precision training
        self.use_amp = config.device.mixed_precision
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Tracking
        self.best_val_f1 = 0.0
        self.best_val_auc = 0.0
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_auc': [],
        }

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(self.train_loader, desc=f'Fold {self.fold_num} - Training')

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits = self.model(input_ids, attention_mask)
                    loss = self.criterion(logits, labels)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )
                self.optimizer.step()

            self.scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        for batch in tqdm(self.val_loader, desc=f'Fold {self.fold_num} - Evaluating'):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            if self.use_amp:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits = self.model(input_ids, attention_mask)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

            total_loss += loss.item()

            # Get predictions and probabilities
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of "matched" class

        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        auc = roc_auc_score(all_labels, all_probs)

        avg_loss = total_loss / len(self.val_loader)

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
        }

        return metrics

    def train(self) -> Dict[str, float]:
        """Train for all epochs with early stopping."""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training Fold {self.fold_num}")
        logger.info(f"{'=' * 60}")

        for epoch in range(self.config.training.num_epochs):
            # Train
            train_loss = self.train_epoch()

            # Evaluate
            val_metrics = self.evaluate()

            # Update history
            self.history['train_loss'].append(train_loss)
            for key, value in val_metrics.items():
                if key in self.history:
                    self.history[f'val_{key}'].append(value)

            # Log metrics
            logger.info(
                f"Epoch {epoch + 1}/{self.config.training.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"Val AUC: {val_metrics['auc']:.4f}"
            )

            # Save best model
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_val_auc = val_metrics['auc']
                self.patience_counter = 0

                # Save checkpoint
                checkpoint = {
                    'fold': self.fold_num,
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_f1': self.best_val_f1,
                    'best_val_auc': self.best_val_auc,
                    'config': OmegaConf.to_container(self.config, resolve=True),
                }
                checkpoint_path = self.output_dir / f'fold_{self.fold_num}_best.pt'
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"✓ Saved best model (F1: {self.best_val_f1:.4f}, AUC: {self.best_val_auc:.4f})")

            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        logger.info(f"\nFold {self.fold_num} training complete!")
        logger.info(f"Best Val F1: {self.best_val_f1:.4f}")
        logger.info(f"Best Val AUC: {self.best_val_auc:.4f}")

        return {
            'fold': self.fold_num,
            'best_val_f1': self.best_val_f1,
            'best_val_auc': self.best_val_auc,
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1],
        }


@hydra.main(version_base=None, config_path="../../conf", config_name="config_nli")
def main(cfg: DictConfig):
    """Main training function with 5-fold CV."""

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Set random seeds
    torch.manual_seed(cfg.data.random_seed)
    np.random.seed(cfg.data.random_seed)

    # Create output directory
    output_dir = Path(cfg.output.base_dir) / cfg.output.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        OmegaConf.save(cfg, f)

    # Create CV splits
    folds_dir = output_dir / 'cv_folds'
    if not (folds_dir / 'nli_cv_folds_metadata.json').exists():
        logger.info("Creating CV folds...")
        create_nli_cv_splits(
            data_dir=cfg.data.data_dir,
            output_dir=str(folds_dir),
            num_folds=cfg.cv.num_folds,
            negative_ratio=cfg.data.negative_ratio,
            use_short_criteria=cfg.data.use_short_criteria,
            random_seed=cfg.data.random_seed,
        )
    else:
        logger.info("Using existing CV folds")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {cfg.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)

    # Train each fold
    fold_results = []

    for fold_num in range(1, cfg.cv.num_folds + 1):
        logger.info(f"\n{'#' * 80}")
        logger.info(f"# FOLD {fold_num}/{cfg.cv.num_folds}")
        logger.info(f"{'#' * 80}\n")

        # Load fold data
        train_dataset, val_dataset = load_nli_fold_split(
            fold_dir=str(folds_dir),
            fold_num=fold_num,
            tokenizer=tokenizer,
            max_length=cfg.data.max_length,
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Create model (fresh model for each fold)
        logger.info(f"Creating model: {cfg.model.name}")
        model = GemmaClassifier(
            num_classes=NUM_CLASSES,  # Binary: 2 classes
            model_name=cfg.model.name,
            pooling_strategy=cfg.model.pooling_strategy,
            freeze_encoder=cfg.model.freeze_encoder,
            hidden_dropout_prob=cfg.model.hidden_dropout_prob,
            classifier_hidden_size=cfg.model.classifier_hidden_size,
            use_gradient_checkpointing=cfg.model.use_gradient_checkpointing,
        )

        # Train fold
        trainer = NLIFoldTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=cfg,
            fold_num=fold_num,
            output_dir=output_dir,
            device='cuda' if cfg.device.use_cuda and torch.cuda.is_available() else 'cpu',
        )

        fold_result = trainer.train()
        fold_results.append(fold_result)

        # Save fold history
        history_path = output_dir / f'fold_{fold_num}_history.json'
        with open(history_path, 'w') as f:
            json.dump(trainer.history, f, indent=2)

    # Aggregate results across folds
    logger.info(f"\n{'=' * 80}")
    logger.info("5-FOLD CROSS-VALIDATION RESULTS")
    logger.info(f"{'=' * 80}\n")

    f1_scores = [r['best_val_f1'] for r in fold_results]
    auc_scores = [r['best_val_auc'] for r in fold_results]

    logger.info("Per-Fold Results:")
    for result in fold_results:
        logger.info(f"  Fold {result['fold']}: F1={result['best_val_f1']:.4f}, AUC={result['best_val_auc']:.4f}")

    logger.info(f"\nAggregate Results:")
    logger.info(f"  Mean F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    logger.info(f"  Mean AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    logger.info(f"  Median F1: {np.median(f1_scores):.4f}")
    logger.info(f"  Median AUC: {np.median(auc_scores):.4f}")

    # Save aggregate results
    aggregate_results = {
        'fold_results': fold_results,
        'mean_f1': float(np.mean(f1_scores)),
        'std_f1': float(np.std(f1_scores)),
        'median_f1': float(np.median(f1_scores)),
        'mean_auc': float(np.mean(auc_scores)),
        'std_auc': float(np.std(auc_scores)),
        'median_auc': float(np.median(auc_scores)),
    }

    results_path = output_dir / 'aggregate_results.json'
    with open(results_path, 'w') as f:
        json.dump(aggregate_results, f, indent=2)

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
