"""
Simple training script for NLI binary classification (single train/val/test split).

For quick testing without 5-fold CV.
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
from tqdm.auto import tqdm
import argparse
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.gemma_encoder import GemmaClassifier
from data.redsm5_nli_dataset import load_redsm5_nli, get_class_weights, NUM_CLASSES
from utils.hardware_optimizer import (
    detect_gpu_info, optimize_pytorch_settings,
    print_hardware_info, compile_model
)


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, use_amp=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for batch in tqdm(dataloader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, use_amp=True):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in tqdm(dataloader, desc='Evaluating'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        total_loss += loss.item()

        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    auc = roc_auc_score(all_labels, all_probs)

    metrics = {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
    }

    return metrics, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='Train NLI binary classifier')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to ReDSM5 data')
    parser.add_argument('--model_name', type=str, default='google/gemma-2-2b', help='Model name')
    parser.add_argument('--pooling', type=str, default='mean', help='Pooling strategy')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='outputs/nli_simple', help='Output directory')
    parser.add_argument('--negative_ratio', type=float, default=1.0, help='Negative to positive ratio')
    parser.add_argument('--use_short_criteria', action='store_true', help='Use short criteria')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder weights')

    args = parser.parse_args()

    # Print hardware information and optimize PyTorch settings
    print_hardware_info()
    optimize_pytorch_settings()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Output: {output_dir}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Load data
    print("\nLoading NLI dataset...")
    train_dataset, val_dataset, test_dataset, pairs_df = load_redsm5_nli(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_length=512,
        negative_ratio=args.negative_ratio,
        use_short_criteria=args.use_short_criteria,
    )

    # Save pairs info
    pairs_df.to_csv(output_dir / 'nli_pairs.csv', index=False)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=4)

    # Create model
    print(f"\nCreating model (binary classification)...")
    model = GemmaClassifier(
        num_classes=NUM_CLASSES,
        model_name=args.model_name,
        pooling_strategy=args.pooling,
        freeze_encoder=args.freeze_encoder,
        hidden_dropout_prob=0.1,
        use_gradient_checkpointing=True,
    ).to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    # Loss with class weights
    class_weights = get_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Class weights: {class_weights.cpu().numpy()}")

    # Training loop
    best_val_f1 = 0.0
    history = {'train_loss': [], 'val_metrics': []}

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_metrics'].append(val_metrics)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")

        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
            }, output_dir / 'best_model.pt')
            print(f"✓ Saved best model (F1: {best_val_f1:.4f})")

    # Load best model and evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")

    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=['unmatched', 'matched']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print(cm)

    # Save results
    results = {
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics,
        'args': vars(args),
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print("Training complete!")


if __name__ == '__main__':
    main()
