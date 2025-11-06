"""
Training script with Hydra configuration and 5-fold cross-validation.

Usage:
    python src/training/train_gemma_hydra.py
    python src/training/train_gemma_hydra.py model.name=google/gemma-2-9b training.batch_size=8
    python src/training/train_gemma_hydra.py cv.num_folds=10
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from pathlib import Path
import json
from tqdm.auto import tqdm
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.gemma_encoder import GemmaClassifier
from data.redsm5_dataset import get_class_weights, NUM_CLASSES
from data.cv_splits import create_cv_splits, load_fold_split, get_fold_statistics


class FoldTrainer:
    """Trainer for a single fold."""

    def __init__(self, cfg: DictConfig, fold_idx: int, device: str = 'cuda'):
        self.cfg = cfg
        self.fold_idx = fold_idx
        self.device = device
        self.best_val_f1 = 0.0

        # Initialize AMP scaler for mixed precision training
        self.use_amp = cfg.device.mixed_precision
        self.scaler = GradScaler() if self.use_amp else None

        # Output directory for this fold
        self.fold_dir = Path(cfg.output.base_dir) / cfg.output.experiment_name / f'fold_{fold_idx}'
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
        }

        print(f"\n{'='*60}")
        print(f"Training Fold {self.fold_idx}")
        print(f"{'='*60}")

        for epoch in range(self.cfg.training.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.cfg.training.num_epochs}")

            train_loss = self.train_epoch(model, train_loader, optimizer, scheduler, criterion)
            val_metrics = self.evaluate(model, val_loader, criterion)

            # Log metrics
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")

            # Save best model
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                checkpoint_path = self.fold_dir / 'best_model.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_metrics': val_metrics,
                    'config': OmegaConf.to_container(self.cfg, resolve=True),
                }, checkpoint_path)
                print(f"✓ Best model saved (F1: {self.best_val_f1:.4f})")

        # Save training history
        with open(self.fold_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        return history, self.best_val_f1


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    """Main training function with 5-fold CV."""

    print("\n" + "="*60)
    print("Gemma Encoder 5-Fold Cross-Validation")
    print("="*60)
    print(OmegaConf.to_yaml(cfg))

    # Setup device
    device = 'cuda' if cfg.device.use_cuda and torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Load tokenizer
    print(f"\nLoading tokenizer: {cfg.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    # Create CV splits
    print(f"\nCreating {cfg.cv.num_folds}-fold cross-validation splits...")
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
    print("\nFold Statistics:")
    print(stats[['fold', 'train_size', 'val_size']])

    # Train each fold
    fold_results = []

    for fold_idx in range(cfg.cv.num_folds):
        print(f"\n{'='*60}")
        print(f"Starting Fold {fold_idx + 1}/{cfg.cv.num_folds}")
        print(f"{'='*60}")

        # Load fold data
        posts_path = Path(cfg.data.data_dir) / 'redsm5_posts.csv'
        train_dataset, val_dataset = load_fold_split(
            data_dir=str(splits_dir),
            fold_idx=fold_idx,
            posts_path=str(posts_path) if posts_path.exists() else None,
            tokenizer=tokenizer,
            max_length=cfg.data.max_length,
        )

        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")

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
        print(f"\nInitializing model: {cfg.model.name}")
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
            print(f"\nUsing class weights: {class_weights.numpy()}")

        # Train fold
        trainer = FoldTrainer(cfg, fold_idx, device)
        history, best_f1 = trainer.train_fold(model, train_loader, val_loader, class_weights)

        fold_results.append({
            'fold': fold_idx,
            'best_val_f1': best_f1,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_val_accuracy': history['val_accuracy'][-1],
        })

        print(f"\nFold {fold_idx} completed. Best F1: {best_f1:.4f}")

    # Aggregate results
    print("\n" + "="*60)
    print("Cross-Validation Results")
    print("="*60)

    results_df = pd.DataFrame(fold_results)
    print(results_df)

    # Compute statistics
    mean_f1 = results_df['best_val_f1'].mean()
    std_f1 = results_df['best_val_f1'].std()

    print(f"\nMean F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Min F1: {results_df['best_val_f1'].min():.4f}")
    print(f"Max F1: {results_df['best_val_f1'].max():.4f}")

    # Save aggregate results
    output_dir = Path(cfg.output.base_dir) / cfg.output.experiment_name
    results_df.to_csv(output_dir / 'cv_results.csv', index=False)

    aggregate_results = {
        'mean_f1': float(mean_f1),
        'std_f1': float(std_f1),
        'min_f1': float(results_df['best_val_f1'].min()),
        'max_f1': float(results_df['best_val_f1'].max()),
        'fold_results': fold_results,
        'config': OmegaConf.to_container(cfg, resolve=True),
    }

    with open(output_dir / 'aggregate_results.json', 'w') as f:
        json.dump(aggregate_results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    import pandas as pd
    main()
