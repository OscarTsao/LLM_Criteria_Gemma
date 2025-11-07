"""
Training script for binary post-criterion matching.

Trains Gemma encoder for binary classification on (post, criterion) pairs.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.gemma_encoder_binary import GemmaBinaryClassifier, load_binary_tokenizer
from src.data.binary_dataset import load_redsm5_binary, get_binary_class_weights
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)


class BinaryTrainer:
    """Trainer for binary classification."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_f1 = 0.0
        self.best_auroc = 0.0

    def train_epoch(self, model, dataloader, optimizer, scheduler, criterion):
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        pbar = tqdm(dataloader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.training.max_grad_norm)
            optimizer.step()
            scheduler.step()

            # Metrics
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class

            # Update progress bar
            if (batch_idx + 1) % self.cfg.logging.log_interval == 0:
                pbar.set_postfix({'loss': loss.item()})

        # Compute epoch metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        auroc = roc_auc_score(all_labels, all_probs)

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auroc': auroc,
        }

    def evaluate(self, model, dataloader, criterion):
        """Evaluate on validation/test set."""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        all_logits = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                total_loss += loss.item()
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_logits.extend(logits.cpu().numpy())

        # Compute metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        auroc = roc_auc_score(all_labels, all_probs)

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auroc': auroc,
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels),
            'probabilities': np.array(all_probs),
            'logits': np.array(all_logits),
        }

    def train(self):
        """Main training loop."""
        cfg = self.cfg

        # Setup logger
        output_dir = Path(cfg.output.base_dir) / cfg.output.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        logger = setup_logger(
            name='train_binary',
            level='INFO',
            log_file=output_dir / 'training.log',
            console=True
        )

        logger.info("="*70)
        logger.info("BINARY POST-CRITERION MATCHING TRAINING")
        logger.info("="*70)
        logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

        # Load tokenizer and data
        logger.info(f"Loading tokenizer: {cfg.model.name}")
        tokenizer = load_binary_tokenizer(cfg.model.name)

        logger.info(f"Loading dataset: {cfg.data.data_dir}")
        train_dataset, val_dataset, test_dataset = load_redsm5_binary(
            data_dir=Path(cfg.data.data_dir),
            tokenizer=tokenizer,
            max_length=cfg.data.max_length,
            negative_sampling=cfg.data.get('negative_sampling', 'all'),
            num_negatives=cfg.data.get('num_negatives', 3),
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=4,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=4,
        )

        # Initialize model
        logger.info("Initializing model...")
        model = GemmaBinaryClassifier(
            model_name=cfg.model.name,
            pooling_strategy=cfg.model.pooling_strategy,
            freeze_encoder=cfg.model.freeze_encoder,
            dropout=cfg.model.get('dropout', 0.1),
        ).to(self.device)

        logger.info(f"Trainable parameters: {model.get_trainable_parameters():,}")
        logger.info(f"Total parameters: {model.get_total_parameters():,}")

        # Loss function with class weights
        if cfg.training.use_class_weights:
            class_weights = get_binary_class_weights(train_dataset).to(self.device)
            logger.info(f"Class weights: {class_weights}")
        else:
            class_weights = None

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

        num_training_steps = len(train_loader) * cfg.training.num_epochs
        num_warmup_steps = int(num_training_steps * cfg.training.warmup_ratio)

        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=num_warmup_steps,
        )

        logger.info(f"Training steps: {num_training_steps}")
        logger.info(f"Warmup steps: {num_warmup_steps}")

        # Training loop
        best_f1 = 0.0
        patience_counter = 0

        for epoch in range(cfg.training.num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{cfg.training.num_epochs}")

            # Train
            train_metrics = self.train_epoch(model, train_loader, optimizer, scheduler, criterion)
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Acc: {train_metrics['accuracy']:.4f}, "
                       f"F1: {train_metrics['f1']:.4f}, "
                       f"AUROC: {train_metrics['auroc']:.4f}")

            # Validate
            val_metrics = self.evaluate(model, val_loader, criterion)
            logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                       f"Acc: {val_metrics['accuracy']:.4f}, "
                       f"F1: {val_metrics['f1']:.4f}, "
                       f"AUROC: {val_metrics['auroc']:.4f}")

            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                patience_counter = 0

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1': best_f1,
                    'config': OmegaConf.to_container(cfg),
                }
                torch.save(checkpoint, output_dir / 'best_model.pt')
                logger.info(f"✓ New best F1: {best_f1:.4f} - Model saved")

                # Save OOF predictions
                np.save(output_dir / 'oof_probs.npy', val_metrics['probabilities'])
                np.save(output_dir / 'oof_labels.npy', val_metrics['labels'])
                np.save(output_dir / 'oof_logits.npy', val_metrics['logits'])

            else:
                patience_counter += 1
                logger.info(f"No improvement ({patience_counter}/{cfg.training.early_stopping_patience})")

                if patience_counter >= cfg.training.early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break

        logger.info("="*70)
        logger.info(f"Training complete! Best F1: {best_f1:.4f}")
        logger.info(f"Model saved to: {output_dir / 'best_model.pt'}")
        logger.info("="*70)


@hydra.main(version_base=None, config_path="../../conf", config_name="config_binary")
def main(cfg: DictConfig):
    """Main entry point."""
    trainer = BinaryTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
