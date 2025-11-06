"""
Training script for Gemma Encoder on ReDSM5 criteria matching task.

Implements best practices from the Gemma Encoder paper:
- Bidirectional attention fine-tuning
- Multiple pooling strategies
- Hyperparameter optimization for dropout rates
- GLUE-style evaluation metrics
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import json
from tqdm.auto import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.gemma_encoder import GemmaClassifier
from data.redsm5_dataset import load_redsm5, get_class_weights, get_symptom_labels, NUM_CLASSES


class GemmaTrainer:
    """Trainer for Gemma encoder on criteria matching."""

    def __init__(
        self,
        model: GemmaClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        num_epochs: int = 10,
        warmup_ratio: float = 0.1,
        device: str = 'cuda',
        class_weights: Optional[torch.Tensor] = None,
        output_dir: str = './outputs',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Loss function with class weights
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.best_val_f1 = 0.0
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_f1': []}

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(self.train_loader, desc='Training')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['symptom_idx'].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Evaluate on validation set."""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0

        for batch in tqdm(self.val_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['symptom_idx'].to(self.device)

            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
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
            'loss': total_loss / len(self.val_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    def train(self):
        """Full training loop."""
        print(f"Starting training for {self.num_epochs} epochs...")

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            train_loss = self.train_epoch()
            val_metrics = self.evaluate()

            # Log metrics
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")

            # Save best model
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.save_checkpoint('best_model.pt', val_metrics)
                print(f"✓ New best model saved (F1: {self.best_val_f1:.4f})")

        # Save training history
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\nTraining complete! Best val F1: {self.best_val_f1:.4f}")

    def save_checkpoint(self, filename: str, metrics: Dict):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }, checkpoint_path)


def main():
    """Main training function."""
    # Configuration
    config = {
        'model_name': 'google/gemma-2-2b',
        'pooling_strategy': 'mean',  # Options: mean, cls, max, attention_kv, attention_query
        'hidden_dropout_prob': 0.1,
        'classifier_hidden_size': 768,
        'learning_rate': 2e-5,
        'batch_size': 16,
        'num_epochs': 10,
        'max_length': 512,
        'data_dir': '/media/cvrlab308/cvrlab308_4090/YuNing/LLM_Criteria_Gemma/data/redsm5',
        'output_dir': './outputs/gemma_criteria',
    }

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    print("Loading dataset...")
    train_dataset, val_dataset, test_dataset = load_redsm5(
        data_dir=config['data_dir'],
        tokenizer=tokenizer,
        max_length=config['max_length']
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

    print("Initializing model...")
    model = GemmaClassifier(
        num_classes=NUM_CLASSES,
        model_name=config['model_name'],
        pooling_strategy=config['pooling_strategy'],
        hidden_dropout_prob=config['hidden_dropout_prob'],
        classifier_hidden_size=config['classifier_hidden_size'],
    )

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Get class weights for imbalance
    class_weights = get_class_weights(train_dataset)
    print(f"\nClass weights: {class_weights}")

    # Create trainer
    trainer = GemmaTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config['learning_rate'],
        num_epochs=config['num_epochs'],
        class_weights=class_weights,
        output_dir=config['output_dir'],
    )

    # Train
    trainer.train()

    print("\nEvaluating on test set...")
    trainer.val_loader = test_loader
    test_metrics = trainer.evaluate()
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")

    # Save config
    with open(Path(config['output_dir']) / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == '__main__':
    main()
