"""
Training script for Binary NLI Criteria Matching Task.

Trains a binary classifier to match posts with DSM-5 criteria.
Input: [CLS] post [SEP] criterion [SEP]
Output: matched (1) or unmatched (0)

Usage:
    python src/training/train_nli_binary.py
    python src/training/train_nli_binary.py model.name=google/gemma-3-4b-it training.batch_size=8
"""

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from tqdm.auto import tqdm
import sys
import mlflow
import mlflow.pytorch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.gemma_encoder import GemmaClassifier
from data.redsm5_nli_dataset import (
    load_redsm5_nli, get_class_weights,
    get_label_distribution, get_criterion_distribution
)
from utils.logger import setup_logger, log_experiment_config

# Binary classification: 2 classes
NUM_CLASSES = 2


class BinaryNLITrainer:
    """Trainer for Binary NLI Criteria Matching."""

    def __init__(self, cfg: DictConfig, run_dir: Path, device: str = 'cuda', logger=None, use_mlflow: bool = False):
        self.cfg = cfg
        self.device = device
        self.run_dir = run_dir
        self.best_val_f1 = float('-inf')
        self.best_epoch = -1
        self.epochs_without_improvement = 0
        self.patience = cfg.training.get('early_stopping_patience')
        self.min_delta = float(cfg.training.get('early_stopping_min_delta', 0.0))
        self.logger = logger
        self.use_mlflow = use_mlflow

        # Mixed precision training
        self.use_amp = cfg.device.mixed_precision

    def train_epoch(self, model, dataloader, optimizer, scheduler, criterion):
        """Train for one epoch."""
        model.train()
        total_loss = 0

        progress_bar = tqdm(dataloader, desc='Training')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            optimizer.zero_grad()

            if self.use_amp:
                with autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.cfg.training.max_grad_norm
                )
                optimizer.step()
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
        """Evaluate on validation/test set."""
        model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0

        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            if self.use_amp:
                with autocast('cuda', dtype=torch.bfloat16):
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

        # Binary classification metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', pos_label=1, zero_division=0
        )

        # Also compute macro average for comparison
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        return {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'confusion_matrix': cm.tolist(),
            'predictions': all_preds,
            'labels': all_labels,
        }

    def train(self, model, train_loader, val_loader, class_weights=None):
        """Train and evaluate the model."""
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
            'val_precision': [],
            'val_recall': [],
            'best_epoch': None,
            'best_val_f1': None,
            'epochs_trained': 0,
        }

        if self.logger:
            self.logger.info("=" * 60)
            self.logger.info("Training Binary NLI Classifier")
            self.logger.info("=" * 60)

        for epoch in range(self.cfg.training.num_epochs):
            if self.logger:
                self.logger.info(f"\nEpoch {epoch + 1}/{self.cfg.training.num_epochs}")

            train_loss = self.train_epoch(model, train_loader, optimizer, scheduler, criterion)
            val_metrics = self.evaluate(model, val_loader, criterion)

            # Log metrics
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])
            history['val_precision'].append(val_metrics['precision'])
            history['val_recall'].append(val_metrics['recall'])

            if self.logger:
                self.logger.info(f"Train Loss: {train_loss:.4f}")
                self.logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
                self.logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
                self.logger.info(f"Val Precision: {val_metrics['precision']:.4f}")
                self.logger.info(f"Val Recall: {val_metrics['recall']:.4f}")
                self.logger.info(f"Val F1 (binary): {val_metrics['f1']:.4f}")
                self.logger.info(f"Val F1 (macro): {val_metrics['f1_macro']:.4f}")

            # Log metrics to MLflow
            if self.use_mlflow:
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                    'val_f1': val_metrics['f1'],
                    'val_f1_macro': val_metrics['f1_macro'],
                    'val_precision_macro': val_metrics['precision_macro'],
                    'val_recall_macro': val_metrics['recall_macro'],
                }, step=epoch)

            # Save best model
            improved = val_metrics['f1'] > (self.best_val_f1 + self.min_delta)
            if improved:
                self.best_val_f1 = val_metrics['f1']
                self.best_epoch = epoch + 1
                self.epochs_without_improvement = 0
                history['best_epoch'] = self.best_epoch
                history['best_val_f1'] = self.best_val_f1

                # Save locally (first part of dual saving)
                checkpoint_path = self.run_dir / 'best_model.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_metrics': val_metrics,
                    'config': OmegaConf.to_container(self.cfg, resolve=True),
                }, checkpoint_path)

                # Log model to MLflow (second part of dual saving)
                if self.use_mlflow:
                    # Create input example for model signature
                    try:
                        sample_batch = next(iter(val_loader))
                        input_example = (
                            sample_batch['input_ids'][:1].to(self.device),
                            sample_batch['attention_mask'][:1].to(self.device)
                        )
                    except:
                        input_example = None

                    # Log the model artifact
                    mlflow.pytorch.log_model(
                        model,
                        name="model",  # Changed from artifact_path to name
                        registered_model_name=None,  # Don't auto-register
                        input_example=input_example,  # Add input example for signature
                    )
                    # Log the checkpoint file as well
                    mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
                    # Log best metrics
                    mlflow.log_metrics({
                        'best_epoch': self.best_epoch,
                        'best_val_f1': self.best_val_f1,
                    })

                if self.logger:
                    self.logger.info(f"✓ Best model saved locally and to MLflow (Epoch {self.best_epoch}, F1: {self.best_val_f1:.4f})" if self.use_mlflow else f"✓ Best model saved (Epoch {self.best_epoch}, F1: {self.best_val_f1:.4f})")
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
        with open(self.run_dir / 'history.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            history_serializable = {k: [float(v) if isinstance(v, (np.floating, np.integer)) else v
                                        for v in val] if isinstance(val, list) else val
                                   for k, val in history.items()}
            json.dump(history_serializable, f, indent=2)

        return history, self.best_val_f1, self.best_epoch


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    """Main training function for Binary NLI task."""

    # Prepare run directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_base = "nli_binary"
    model_label = cfg.model.name.replace("/", "_")
    output_root = Path(to_absolute_path(cfg.output.base_dir))
    run_name = f"{experiment_base}-{model_label}-{timestamp}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger(
        name='train_nli_binary',
        level='INFO',
        log_file=run_dir / 'training.log',
        console=True,
    )

    logger.info("=" * 60)
    logger.info("Binary NLI Criteria Matching Training")
    logger.info("=" * 60)
    logger.info("Task: Match posts with DSM-5 criteria (binary classification)")
    logger.info("Input format: [CLS] post [SEP] criterion [SEP]")
    logger.info("Output: matched (1) or unmatched (0)")

    # Log configuration
    log_experiment_config(logger, OmegaConf.to_container(cfg, resolve=True))

    # Setup device
    device = 'cuda' if cfg.device.use_cuda and torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    logger.info(f"Run directory: {run_dir}")

    # Save configuration
    with open(run_dir / 'config.yaml', 'w') as config_file:
        config_file.write(OmegaConf.to_yaml(cfg))

    # Load tokenizer
    logger.info(f"Loading tokenizer: {cfg.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    # Load NLI dataset
    logger.info("Loading ReDSM5 NLI dataset...")
    data_dir = to_absolute_path(cfg.data.data_dir)
    train_dataset, val_dataset, test_dataset = load_redsm5_nli(
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=cfg.data.max_length,
        test_size=cfg.data.test_size,
        val_size=cfg.data.val_size,
        random_seed=cfg.data.random_seed,
    )

    # Show dataset statistics
    logger.info("\nDataset Statistics:")
    logger.info(f"Train: {len(train_dataset)} pairs")
    logger.info(f"Val: {len(val_dataset)} pairs")
    logger.info(f"Test: {len(test_dataset)} pairs")

    # Show label distribution
    logger.info("\nLabel Distribution (Train):")
    train_dist = get_label_distribution(train_dataset)
    logger.info(f"\n{train_dist}")

    # Show criterion distribution
    logger.info("\nCriterion Distribution (Train):")
    criterion_dist = get_criterion_distribution(train_dataset)
    logger.info(f"\n{criterion_dist}")

    # Save distributions
    train_dist.to_csv(run_dir / 'train_label_distribution.csv', index=False)
    criterion_dist.to_csv(run_dir / 'train_criterion_distribution.csv', index=False)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.get('num_workers', 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.get('num_workers', 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.get('num_workers', 0),
    )

    # Initialize model for BINARY classification
    logger.info(f"\nInitializing binary classifier: {cfg.model.name}")
    logger.info(f"Number of classes: {NUM_CLASSES} (matched/unmatched)")

    model = GemmaClassifier(
        num_classes=NUM_CLASSES,  # Binary classification
        model_name=cfg.model.name,
        pooling_strategy=cfg.model.pooling_strategy,
        freeze_encoder=cfg.model.freeze_encoder,
        hidden_dropout_prob=cfg.model.hidden_dropout_prob,
        classifier_hidden_size=cfg.model.classifier_hidden_size,
        device=device,
        use_gradient_checkpointing=cfg.model.get('use_gradient_checkpointing', False),
        use_dora=cfg.model.get('use_dora', True),
        dora_rank=cfg.model.get('dora_rank', 16),
        dora_alpha=cfg.model.get('dora_alpha', 32.0),
        dora_dropout=cfg.model.get('dora_dropout', 0.05),
    )

    # Get class weights
    class_weights = None
    if cfg.training.use_class_weights:
        class_weights = get_class_weights(train_dataset)
        logger.info(f"\nUsing class weights: {class_weights.numpy()}")

    # Setup MLflow tracking
    use_mlflow = cfg.mlflow.get('enabled', False)
    if use_mlflow:
        # Set tracking URI and artifact location
        mlflow.set_tracking_uri(to_absolute_path(cfg.mlflow.tracking_uri))

        # Create or get experiment
        experiment_name = cfg.mlflow.experiment_name
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=to_absolute_path(cfg.mlflow.artifact_location)
            )
        else:
            experiment_id = experiment.experiment_id

        # Start MLflow run
        run_name = cfg.mlflow.get('run_name') or run_name
        mlflow.start_run(experiment_id=experiment_id, run_name=run_name)

        # Log all configuration parameters
        mlflow.log_params({
            'model_name': cfg.model.name,
            'pooling_strategy': cfg.model.pooling_strategy,
            'batch_size': cfg.training.batch_size,
            'learning_rate': cfg.training.learning_rate,
            'num_epochs': cfg.training.num_epochs,
            'weight_decay': cfg.training.weight_decay,
            'warmup_ratio': cfg.training.warmup_ratio,
            'max_grad_norm': cfg.training.max_grad_norm,
            'use_class_weights': cfg.training.use_class_weights,
            'early_stopping_patience': cfg.training.early_stopping_patience,
            'max_length': cfg.data.max_length,
            'test_size': cfg.data.test_size,
            'val_size': cfg.data.val_size,
            'use_dora': cfg.model.get('use_dora', True),
            'dora_rank': cfg.model.get('dora_rank', 16),
            'dora_alpha': cfg.model.get('dora_alpha', 32.0),
            'mixed_precision': cfg.device.mixed_precision,
            'gradient_checkpointing': cfg.model.get('use_gradient_checkpointing', False),
        })

        # Log dataset statistics
        mlflow.log_metrics({
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'test_size': len(test_dataset),
        })

        # Log tags
        if 'tags' in cfg.mlflow:
            mlflow.set_tags(OmegaConf.to_container(cfg.mlflow.tags, resolve=True))

        logger.info(f"MLflow tracking enabled: {mlflow.get_tracking_uri()}")
        logger.info(f"MLflow experiment: {experiment_name}")
        logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    # Train model
    trainer = BinaryNLITrainer(cfg, run_dir, device, logger=logger, use_mlflow=use_mlflow)
    history, best_f1, best_epoch = trainer.train(model, train_loader, val_loader, class_weights)

    # Evaluate on test set
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating on Test Set")
    logger.info("=" * 60)

    # Load best model
    checkpoint = torch.load(run_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    test_metrics = trainer.evaluate(model, test_loader, criterion)

    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
    logger.info(f"Test F1 (binary): {test_metrics['f1']:.4f}")
    logger.info(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")

    # Confusion matrix
    cm = np.array(test_metrics['confusion_matrix'])
    logger.info("\nConfusion Matrix:")
    logger.info("                 Predicted")
    logger.info("               Unmatched  Matched")
    logger.info(f"Actual Unmatched  {cm[0,0]:6d}   {cm[0,1]:6d}")
    logger.info(f"Actual Matched    {cm[1,0]:6d}   {cm[1,1]:6d}")

    # Save test results
    test_results = {
        'test_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer, float)) else v
                        for k, v in test_metrics.items() if k not in ['predictions', 'labels']},
        'best_epoch': int(best_epoch),
        'best_val_f1': float(best_f1),
        'config': OmegaConf.to_container(cfg, resolve=True),
    }

    with open(run_dir / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)

    # Per-criterion analysis
    logger.info("\n" + "=" * 60)
    logger.info("Per-Criterion Test Performance")
    logger.info("=" * 60)

    criterion_results = {}
    for criterion_name in set(test_dataset.criterion_names):
        # Get indices for this criterion
        indices = [i for i, c in enumerate(test_dataset.criterion_names) if c == criterion_name]

        # Extract predictions and labels for this criterion
        preds = [test_metrics['predictions'][i] for i in indices]
        labels = [test_metrics['labels'][i] for i in indices]

        # Compute metrics
        acc = accuracy_score(labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', pos_label=1, zero_division=0
        )

        criterion_results[criterion_name] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'num_samples': len(indices),
            'num_matched': sum(labels),
        }

    # Log and save per-criterion results
    criterion_df = pd.DataFrame(criterion_results).T
    criterion_df = criterion_df.sort_values('f1', ascending=False)
    logger.info(f"\n{criterion_df}")
    criterion_df.to_csv(run_dir / 'per_criterion_results.csv')

    # Log test metrics and artifacts to MLflow
    if use_mlflow:
        # Log overall test metrics
        mlflow.log_metrics({
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1'],
            'test_f1_macro': test_metrics['f1_macro'],
            'test_precision_macro': test_metrics['precision_macro'],
            'test_recall_macro': test_metrics['recall_macro'],
        })

        # Log per-criterion test metrics
        for criterion_name, metrics in criterion_results.items():
            mlflow.log_metrics({
                f'test_{criterion_name}_accuracy': metrics['accuracy'],
                f'test_{criterion_name}_precision': metrics['precision'],
                f'test_{criterion_name}_recall': metrics['recall'],
                f'test_{criterion_name}_f1': metrics['f1'],
            })

        # Log artifacts (results files)
        mlflow.log_artifact(str(run_dir / 'test_results.json'), artifact_path="results")
        mlflow.log_artifact(str(run_dir / 'per_criterion_results.csv'), artifact_path="results")
        mlflow.log_artifact(str(run_dir / 'training.log'), artifact_path="logs")
        mlflow.log_artifact(str(run_dir / 'config.yaml'), artifact_path="config")

        # End MLflow run
        mlflow.end_run()
        logger.info("MLflow run completed and closed")

    logger.info(f"\nResults saved to: {run_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
