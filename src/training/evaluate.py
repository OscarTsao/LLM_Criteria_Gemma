"""
Evaluation script for Gemma Encoder on ReDSM5.

Implements GLUE-style metrics as specified in the paper.
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import numpy as np
from pathlib import Path
import json
import argparse
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.gemma_encoder import GemmaClassifier
from data.redsm5_dataset import load_redsm5, get_symptom_labels, NUM_CLASSES


@torch.no_grad()
def evaluate_model(
    model: GemmaClassifier,
    dataloader: DataLoader,
    device: str = 'cuda'
) -> Dict:
    """
    Evaluate model on dataset.

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['symptom_idx']

        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=-1)
        preds = torch.argmax(outputs, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    conf_matrix = confusion_matrix(all_labels, all_preds)

    symptom_labels = get_symptom_labels()
    per_class_metrics = {}
    for i, label in enumerate(symptom_labels):
        per_class_metrics[label] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }

    results = {
        'accuracy': float(accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'per_class': per_class_metrics,
        'confusion_matrix': conf_matrix.tolist(),
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
        'probabilities': all_probs.tolist(),
    }

    return results


def plot_confusion_matrix(conf_matrix: np.ndarray, output_path: Path):
    """Plot confusion matrix."""
    labels = get_symptom_labels()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def print_results(results: Dict):
    """Print formatted results."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:   {results['accuracy']:.4f}")
    print(f"  Macro P:    {results['macro_precision']:.4f}")
    print(f"  Macro R:    {results['macro_recall']:.4f}")
    print(f"  Macro F1:   {results['macro_f1']:.4f}")

    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 60)

    for label, metrics in results['per_class'].items():
        print(f"{label:<20} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
              f"{metrics['f1']:>10.4f} {metrics['support']:>10}")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Gemma Encoder on ReDSM5')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str,
                        default='/media/cvrlab308/cvrlab308_4090/YuNing/LLM_Criteria_Gemma/data/redsm5',
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='./eval_results', help='Output directory')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Initialize model
    model = GemmaClassifier(
        num_classes=NUM_CLASSES,
        model_name='google/gemma-2-2b',  # Adjust if needed
        pooling_strategy='mean'
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    print("Loading dataset...")
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b')
    train_dataset, val_dataset, test_dataset = load_redsm5(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_length=512
    )

    # Select dataset split
    dataset_map = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    dataset = dataset_map[args.split]
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    print(f"Evaluating on {args.split} set ({len(dataset)} samples)...")
    results = evaluate_model(model, dataloader, device)

    # Print results
    print_results(results)

    # Save results
    results_file = output_dir / f'{args.split}_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Plot confusion matrix
    conf_matrix = np.array(results['confusion_matrix'])
    plot_path = output_dir / f'{args.split}_confusion_matrix.png'
    plot_confusion_matrix(conf_matrix, plot_path)
    print(f"Confusion matrix saved to: {plot_path}")


if __name__ == '__main__':
    main()
