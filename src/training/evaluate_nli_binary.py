"""
Evaluation script for Binary NLI Criteria Matching Model.

Evaluates a trained binary classifier on the test set and generates
detailed performance reports including per-criterion analysis.

Usage:
    python src/training/evaluate_nli_binary.py --checkpoint outputs/nli_binary-google_gemma-3-4b-it-20241117/best_model.pt
    python src/training/evaluate_nli_binary.py --checkpoint <path> --output-dir <path>
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.gemma_encoder import GemmaClassifier
from data.redsm5_nli_dataset import load_redsm5_nli, get_criterion_distribution
from data.dsm5_criteria import get_symptom_labels


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a dataset.

    Returns:
        dict with predictions, labels, logits, post_ids, and criterion_names
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    all_post_ids = []
    all_criterion_names = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            logits = model(input_ids, attention_mask)

            # Get predictions
            preds = torch.argmax(logits, dim=-1)

            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
            all_post_ids.extend(batch['post_id'])
            all_criterion_names.extend(batch['criterion_name'])

    return {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'logits': np.array(all_logits),
        'post_ids': all_post_ids,
        'criterion_names': all_criterion_names,
    }


def compute_metrics(labels, predictions, logits=None):
    """Compute comprehensive evaluation metrics."""
    metrics = {}

    # Accuracy
    metrics['accuracy'] = accuracy_score(labels, predictions)

    # Binary metrics (positive class = 1 = matched)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', pos_label=1, zero_division=0
    )
    metrics['precision'] = prec
    metrics['recall'] = rec
    metrics['f1'] = f1

    # Macro metrics
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    metrics['precision_macro'] = prec_macro
    metrics['recall_macro'] = rec_macro
    metrics['f1_macro'] = f1_macro

    # Per-class metrics
    prec_per_class, rec_per_class, f1_per_class, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    metrics['precision_per_class'] = prec_per_class.tolist()
    metrics['recall_per_class'] = rec_per_class.tolist()
    metrics['f1_per_class'] = f1_per_class.tolist()
    metrics['support_per_class'] = support.tolist()

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    metrics['confusion_matrix'] = cm.tolist()

    # AUC if logits provided
    if logits is not None:
        # Get probability of positive class
        probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        try:
            auc = roc_auc_score(labels, probs)
            metrics['auc'] = auc
            metrics['probabilities'] = probs.tolist()
        except:
            metrics['auc'] = None

    return metrics


def per_criterion_analysis(results, symptom_labels):
    """Analyze performance per criterion."""
    criterion_metrics = {}

    for criterion_name in symptom_labels:
        # Get indices for this criterion
        indices = [i for i, c in enumerate(results['criterion_names']) if c == criterion_name]

        if not indices:
            continue

        # Extract data for this criterion
        labels = results['labels'][indices]
        preds = results['predictions'][indices]
        logits = results['logits'][indices]

        # Compute metrics
        metrics = compute_metrics(labels, preds, logits)

        criterion_metrics[criterion_name] = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'auc': metrics.get('auc'),
            'num_samples': len(indices),
            'num_matched': int(labels.sum()),
            'num_unmatched': int((1 - labels).sum()),
            'confusion_matrix': metrics['confusion_matrix'],
        }

    return criterion_metrics


def save_confusion_matrix_plot(cm, output_path):
    """Save confusion matrix as a heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Unmatched', 'Matched'],
        yticklabels=['Unmatched', 'Matched']
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_per_criterion_plot(criterion_df, output_path):
    """Save per-criterion performance plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # F1 scores
    criterion_df_sorted = criterion_df.sort_values('f1', ascending=True)
    axes[0, 0].barh(range(len(criterion_df_sorted)), criterion_df_sorted['f1'])
    axes[0, 0].set_yticks(range(len(criterion_df_sorted)))
    axes[0, 0].set_yticklabels(criterion_df_sorted.index)
    axes[0, 0].set_xlabel('F1 Score')
    axes[0, 0].set_title('F1 Score by Criterion')
    axes[0, 0].grid(axis='x', alpha=0.3)

    # Precision and Recall
    x = np.arange(len(criterion_df))
    width = 0.35
    axes[0, 1].bar(x - width/2, criterion_df['precision'], width, label='Precision')
    axes[0, 1].bar(x + width/2, criterion_df['recall'], width, label='Recall')
    axes[0, 1].set_xlabel('Criterion')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Precision and Recall by Criterion')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(criterion_df.index, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Sample distribution
    axes[1, 0].bar(range(len(criterion_df)), criterion_df['num_matched'], label='Matched')
    axes[1, 0].bar(range(len(criterion_df)), criterion_df['num_unmatched'],
                   bottom=criterion_df['num_matched'], label='Unmatched')
    axes[1, 0].set_xlabel('Criterion')
    axes[1, 0].set_ylabel('Number of Samples')
    axes[1, 0].set_title('Sample Distribution by Criterion')
    axes[1, 0].set_xticks(range(len(criterion_df)))
    axes[1, 0].set_xticklabels(criterion_df.index, rotation=45, ha='right')
    axes[1, 0].legend()

    # Accuracy
    axes[1, 1].bar(range(len(criterion_df)), criterion_df['accuracy'])
    axes[1, 1].set_xlabel('Criterion')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Accuracy by Criterion')
    axes[1, 1].set_xticks(range(len(criterion_df)))
    axes[1, 1].set_xticklabels(criterion_df.index, rotation=45, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Binary NLI Criteria Matching Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pt file)')
    parser.add_argument('--data-dir', type=str, default='data/redsm5',
                       help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (default: checkpoint directory)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length')
    args = parser.parse_args()

    # Setup output directory
    checkpoint_path = Path(args.checkpoint)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Binary NLI Criteria Matching - Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")

    # Load checkpoint
    print("\nLoading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', {})

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load tokenizer
    model_name = config.get('model', {}).get('name', 'google/gemma-3-4b-it')
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load dataset
    print("\nLoading test dataset...")
    _, _, test_dataset = load_redsm5_nli(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
        test_size=config.get('data', {}).get('test_size', 0.15),
        val_size=config.get('data', {}).get('val_size', 0.15),
        random_seed=config.get('data', {}).get('random_seed', 42),
    )

    print(f"Test set: {len(test_dataset)} pairs")

    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # Initialize model
    print("\nInitializing model...")
    model = GemmaClassifier(
        num_classes=2,  # Binary classification
        model_name=model_name,
        pooling_strategy=config.get('model', {}).get('pooling_strategy', 'mean'),
        hidden_dropout_prob=config.get('model', {}).get('hidden_dropout_prob', 0.1),
        classifier_hidden_size=config.get('model', {}).get('classifier_hidden_size'),
        device=device,
    )

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Model loaded successfully")

    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)

    results = evaluate_model(model, test_loader, device)

    # Compute overall metrics
    print("\nComputing metrics...")
    overall_metrics = compute_metrics(
        results['labels'],
        results['predictions'],
        results['logits']
    )

    # Print overall metrics
    print("\n" + "=" * 60)
    print("Overall Test Results")
    print("=" * 60)
    print(f"Accuracy: {overall_metrics['accuracy']:.4f}")
    print(f"Precision: {overall_metrics['precision']:.4f}")
    print(f"Recall: {overall_metrics['recall']:.4f}")
    print(f"F1 Score (binary): {overall_metrics['f1']:.4f}")
    print(f"F1 Score (macro): {overall_metrics['f1_macro']:.4f}")
    if overall_metrics.get('auc'):
        print(f"AUC: {overall_metrics['auc']:.4f}")

    # Print confusion matrix
    cm = np.array(overall_metrics['confusion_matrix'])
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("               Unmatched  Matched")
    print(f"Actual Unmatched  {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"Actual Matched    {cm[1,0]:6d}   {cm[1,1]:6d}")

    # Per-criterion analysis
    print("\n" + "=" * 60)
    print("Per-Criterion Analysis")
    print("=" * 60)

    symptom_labels = get_symptom_labels()
    criterion_metrics = per_criterion_analysis(results, symptom_labels)

    criterion_df = pd.DataFrame(criterion_metrics).T
    criterion_df = criterion_df.sort_values('f1', ascending=False)
    print(f"\n{criterion_df[['accuracy', 'precision', 'recall', 'f1', 'num_matched', 'num_samples']]}")

    # Save results
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)

    # Save overall metrics
    with open(output_dir / 'evaluation_metrics.json', 'w') as f:
        # Remove non-serializable fields
        metrics_to_save = {k: v for k, v in overall_metrics.items()
                          if k not in ['probabilities']}
        json.dump(metrics_to_save, f, indent=2)
    print(f"Saved: {output_dir / 'evaluation_metrics.json'}")

    # Save per-criterion metrics
    criterion_df.to_csv(output_dir / 'per_criterion_metrics.csv')
    print(f"Saved: {output_dir / 'per_criterion_metrics.csv'}")

    # Save detailed per-criterion metrics
    with open(output_dir / 'per_criterion_metrics.json', 'w') as f:
        json.dump(criterion_metrics, f, indent=2)
    print(f"Saved: {output_dir / 'per_criterion_metrics.json'}")

    # Save confusion matrix plot
    save_confusion_matrix_plot(cm, output_dir / 'confusion_matrix.png')
    print(f"Saved: {output_dir / 'confusion_matrix.png'}")

    # Save per-criterion plot
    save_per_criterion_plot(criterion_df, output_dir / 'per_criterion_performance.png')
    print(f"Saved: {output_dir / 'per_criterion_performance.png'}")

    # Save classification report
    target_names = ['Unmatched', 'Matched']
    report = classification_report(
        results['labels'],
        results['predictions'],
        target_names=target_names
    )
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write(report)
    print(f"Saved: {output_dir / 'classification_report.txt'}")

    print("\n" + "=" * 60)
    print("Evaluation Complete")
    print("=" * 60)


if __name__ == '__main__':
    main()
