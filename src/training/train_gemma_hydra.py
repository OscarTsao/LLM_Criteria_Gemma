"""
Training script with Hydra configuration and 5-fold cross-validation.

Usage:
    python src/training/train_gemma_hydra.py
    python src/training/train_gemma_hydra.py model.name=google/gemma-2-9b training.batch_size=8
    python src/training/train_gemma_hydra.py cv.num_folds=10
"""

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from tqdm.auto import tqdm
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.gemma_encoder import GemmaClassifier
from data.redsm5_dataset import get_class_weights, NUM_CLASSES
from data.cv_splits import create_cv_splits, load_fold_split, get_fold_statistics
from utils.logger import setup_logger, log_experiment_config, log_training_summary


class FoldTrainer:
    """Trainer for a single fold."""

    def __init__(self, cfg: DictConfig, fold_idx: int, run_dir: Path, device: str = 'cuda', logger=None):
        self.cfg = cfg
        self.fold_idx = fold_idx
        self.device = device
        self.run_dir = run_dir
        self.best_val_f1 = float('-inf')
        self.best_epoch = -1
        self.epochs_without_improvement = 0
        self.patience = cfg.training.get('early_stopping_patience')
        self.min_delta = float(cfg.training.get('early_stopping_min_delta', 0.0))
        self.logger = logger

        # Initialize AMP scaler for mixed precision training
        self.use_amp = cfg.device.mixed_precision
        self.scaler = GradScaler() if self.use_amp else None

        # Output directory for this fold
        self.fold_dir = run_dir / f'fold_{fold_idx}'
        self.fold_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, model, dataloader, optimizer, scheduler, criterion):
        """Train for one epoch with AMP support."""
        model.train()
        total_loss = 0

        progress_bar = tqdm(dataloader, desc=f'Fold {self.fold_idx} - Training')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['symptom_idx'].to(self.device)

            optimizer.zero_grad()

            # Use automatic mixed precision if enabled
            if self.use_amp:
                with autocast(dtype=torch.bfloat16):
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)

                # Scaled backward pass
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.cfg.training.max_grad_norm
                )
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.cfg.training.max_grad_norm
                )
                optimizer.step()

            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(dataloader)

    @torch.no_grad()
    def evaluate(self, model, dataloader, criterion):
        """Evaluate on validation set with AMP support."""
        model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0

        for batch in tqdm(dataloader, desc=f'Fold {self.fold_idx} - Evaluating'):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['symptom_idx'].to(self.device)

            # Use autocast for evaluation too (saves memory)
            if self.use_amp:
                with autocast(dtype=torch.bfloat16):
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )

        return {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    def train_fold(self, model, train_loader, val_loader, class_weights=None):
        """Train and evaluate a single fold."""
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay
        )

        total_steps = len(train_loader) * self.cfg.training.num_epochs
        warmup_steps = int(total_steps * self.cfg.training.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Loss function
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'best_epoch': None,
            'best_val_f1': None,
            'epochs_trained': 0,
        }

        if self.logger:
            self.logger.info("=" * 60)
            self.logger.info(f"Training Fold {self.fold_idx}")
            self.logger.info("=" * 60)

        for epoch in range(self.cfg.training.num_epochs):
            if self.logger:
                self.logger.info(f"Epoch {epoch + 1}/{self.cfg.training.num_epochs}")

            train_loss = self.train_epoch(model, train_loader, optimizer, scheduler, criterion)
            val_metrics = self.evaluate(model, val_loader, criterion)

            # Log metrics
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])

            if self.logger:
                log_training_summary(
                    self.logger,
                    fold=self.fold_idx,
                    epoch=epoch + 1,
                    train_loss=train_loss,
                    val_metrics=val_metrics,
                    is_best=False,
                )

            # Save best model
            improved = val_metrics['f1'] > (self.best_val_f1 + self.min_delta)
            if improved:
                self.best_val_f1 = val_metrics['f1']
                self.best_epoch = epoch + 1
                self.epochs_without_improvement = 0
                history['best_epoch'] = self.best_epoch
                history['best_val_f1'] = self.best_val_f1
                checkpoint_path = self.fold_dir / 'best_model.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_metrics': val_metrics,
                    'config': OmegaConf.to_container(self.cfg, resolve=True),
                }, checkpoint_path)
                if self.logger:
                    self.logger.info(f"✓ Best model saved (Epoch {self.best_epoch}, F1: {self.best_val_f1:.4f})")
            else:
                self.epochs_without_improvement += 1
                if self.patience and self.epochs_without_improvement >= self.patience:
                    if self.logger:
                        self.logger.warning(
                            f"✗ Early stopping triggered after {epoch + 1} epochs "
                            f"(no F1 improvement for {self.patience} epochs)."
                        )
                    break

        history['epochs_trained'] = len(history['train_loss'])
        if history['best_epoch'] is None:
            history['best_epoch'] = history['epochs_trained']
            history['best_val_f1'] = self.best_val_f1

        # Save training history
        with open(self.fold_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        return history, self.best_val_f1, self.best_epoch


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    """Main training function with 5-fold CV."""

    # Prepare run directory with timestamp and model signature
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_base = str(cfg.output.experiment_name or "experiment").replace(" ", "_")
    model_label = cfg.model.name.replace("/", "_")
    output_root = Path(to_absolute_path(cfg.output.base_dir))
    run_name = f"{experiment_base}-{model_label}-{timestamp}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger(
        name='train_gemma_hydra',
        level='INFO',
        log_file=run_dir / 'training.log',
        console=True,
    )

    logger.info("=" * 60)
    logger.info("Gemma Encoder 5-Fold Cross-Validation")
    logger.info("=" * 60)

    # Log configuration
    log_experiment_config(logger, OmegaConf.to_container(cfg, resolve=True))

    # Setup device
    device = 'cuda' if cfg.device.use_cuda and torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    logger.info(f"Run directory: {run_dir}")

    # Persist resolved Hydra configuration for reproducibility
    with open(run_dir / 'config.yaml', 'w') as config_file:
        config_file.write(OmegaConf.to_yaml(cfg))

    # Load tokenizer
    logger.info(f"Loading tokenizer: {cfg.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    # Create CV splits
    logger.info(f"Creating {cfg.cv.num_folds}-fold cross-validation splits...")
    splits_dir = Path(cfg.data.data_dir) / 'cv_splits'

    annotations_path = Path(cfg.data.data_dir) / 'redsm5_annotations.csv'
    splits = create_cv_splits(
        annotations_path=str(annotations_path),
        num_folds=cfg.cv.num_folds,
        random_seed=cfg.data.random_seed,
        output_dir=str(splits_dir) if cfg.cv.save_fold_results else None,
    )

    # Print fold statistics
    stats = get_fold_statistics(splits)
    logger.info("Fold Statistics:")
    logger.info(f"\n{stats[['fold', 'train_size', 'val_size']]}")

    # Train each fold
    fold_results = []

    for fold_idx in range(cfg.cv.num_folds):
        logger.info("=" * 60)
        logger.info(f"Starting Fold {fold_idx + 1}/{cfg.cv.num_folds}")
        logger.info("=" * 60)

        # Load fold data
        posts_path = Path(cfg.data.data_dir) / 'redsm5_posts.csv'
        train_dataset, val_dataset = load_fold_split(
            data_dir=str(splits_dir),
            fold_idx=fold_idx,
            posts_path=str(posts_path) if posts_path.exists() else None,
            tokenizer=tokenizer,
            max_length=cfg.data.max_length,
        )

        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size
        )

        # Initialize model
        logger.info(f"Initializing model: {cfg.model.name}")
        model = GemmaClassifier(
            num_classes=NUM_CLASSES,
            model_name=cfg.model.name,
            pooling_strategy=cfg.model.pooling_strategy,
            freeze_encoder=cfg.model.freeze_encoder,
            hidden_dropout_prob=cfg.model.hidden_dropout_prob,
            classifier_hidden_size=cfg.model.classifier_hidden_size,
            device=device,
            use_gradient_checkpointing=cfg.model.get('use_gradient_checkpointing', False),
        )

        # Get class weights
        class_weights = None
        if cfg.training.use_class_weights:
            class_weights = get_class_weights(train_dataset)
            logger.info(f"Using class weights: {class_weights.numpy()}")

        # Train fold
        trainer = FoldTrainer(cfg, fold_idx, run_dir, device, logger=logger)
        history, best_f1, best_epoch = trainer.train_fold(model, train_loader, val_loader, class_weights)

        fold_results.append({
            'fold': fold_idx,
            'best_val_f1': best_f1,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_val_accuracy': history['val_accuracy'][-1],
            'best_epoch': best_epoch,
            'epochs_trained': history['epochs_trained'],
        })

        logger.info(f"Fold {fold_idx} completed. Best F1: {best_f1:.4f}")

    # Aggregate results
    logger.info("=" * 60)
    logger.info("Cross-Validation Results")
    logger.info("=" * 60)

    results_df = pd.DataFrame(fold_results)
    logger.info(f"\n{results_df}")

    # Compute statistics
    mean_f1 = results_df['best_val_f1'].mean()
    std_f1 = results_df['best_val_f1'].std()

    logger.info(f"Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
    logger.info(f"Min F1: {results_df['best_val_f1'].min():.4f}")
    logger.info(f"Max F1: {results_df['best_val_f1'].max():.4f}")

    # Save aggregate results
    results_df.to_csv(run_dir / 'cv_results.csv', index=False)

    aggregate_results = {
        'mean_f1': float(mean_f1),
        'std_f1': float(std_f1),
        'min_f1': float(results_df['best_val_f1'].min()),
        'max_f1': float(results_df['best_val_f1'].max()),
        'fold_results': fold_results,
        'config': OmegaConf.to_container(cfg, resolve=True),
        'run_metadata': {
            'timestamp': timestamp,
            'run_name': run_name,
            'model_name': cfg.model.name,
            'experiment_name': cfg.output.experiment_name,
            'output_directory': str(run_dir),
        },
    }

    with open(run_dir / 'aggregate_results.json', 'w') as f:
        json.dump(aggregate_results, f, indent=2)

    logger.info(f"Results saved to: {run_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
