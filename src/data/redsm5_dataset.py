"""
ReDSM5 Dataset Loader

Loads and processes the ReDSM5 (Reddit DSM-5) dataset for depression symptom classification.
Dataset contains Reddit posts annotated for 9 DSM-5 depression symptoms plus a special case.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple, List
from pathlib import Path
import numpy as np


# 10 classes total: 9 DSM-5 symptoms + 1 special case
NUM_CLASSES = 10

# DSM-5 Depression Symptom Labels
SYMPTOM_LABELS = [
    'DEPRESSED_MOOD',
    'ANHEDONIA',
    'APPETITE_CHANGE',
    'SLEEP_ISSUES',
    'PSYCHOMOTOR',
    'FATIGUE',
    'WORTHLESSNESS',
    'COGNITIVE_ISSUES',
    'SUICIDAL_THOUGHTS',
    'SPECIAL_CASE',
]


class ReDSM5Dataset(Dataset):
    """PyTorch Dataset for ReDSM5."""

    def __init__(
        self,
        texts: List[str],
        symptom_indices: List[int],
        tokenizer,
        max_length: int = 512,
    ):
        """
        Initialize ReDSM5 Dataset.

        Args:
            texts: List of text strings (Reddit posts)
            symptom_indices: List of symptom class indices (0-9)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.texts = texts
        self.symptom_indices = symptom_indices
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        symptom_idx = self.symptom_indices[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'symptom_idx': torch.tensor(symptom_idx, dtype=torch.long),
        }


def load_redsm5(
    data_dir: str,
    tokenizer,
    max_length: int = 512,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42,
) -> Tuple[ReDSM5Dataset, ReDSM5Dataset, ReDSM5Dataset]:
    """
    Load ReDSM5 dataset and create train/val/test splits.

    Args:
        data_dir: Path to directory containing redsm5_posts.csv and redsm5_annotations.csv
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining after test split)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_path = Path(data_dir)

    # Load posts and annotations
    posts_df = pd.read_csv(data_path / 'redsm5_posts.csv')
    annotations_df = pd.read_csv(data_path / 'redsm5_annotations.csv')

    # Merge posts with annotations
    # Assuming annotations_df has columns: post_id, symptom_label
    # and posts_df has columns: post_id, text
    merged_df = pd.merge(annotations_df, posts_df, on='post_id', how='inner')

    # Convert symptom labels to indices
    symptom_to_idx = {label: idx for idx, label in enumerate(SYMPTOM_LABELS)}
    merged_df['symptom_idx'] = merged_df['symptom_label'].map(symptom_to_idx)

    # Handle any unmapped symptoms
    if merged_df['symptom_idx'].isna().any():
        print(f"Warning: {merged_df['symptom_idx'].isna().sum()} unmapped symptoms found")
        merged_df = merged_df.dropna(subset=['symptom_idx'])

    merged_df['symptom_idx'] = merged_df['symptom_idx'].astype(int)

    texts = merged_df['text'].tolist()
    symptom_indices = merged_df['symptom_idx'].tolist()

    # Stratified train/test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts,
        symptom_indices,
        test_size=test_size,
        random_state=random_seed,
        stratify=symptom_indices,
    )

    # Stratified train/val split
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio,
        random_state=random_seed,
        stratify=y_temp,
    )

    # Create datasets
    train_dataset = ReDSM5Dataset(X_train, y_train, tokenizer, max_length)
    val_dataset = ReDSM5Dataset(X_val, y_val, tokenizer, max_length)
    test_dataset = ReDSM5Dataset(X_test, y_test, tokenizer, max_length)

    print(f"Dataset loaded successfully:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    return train_dataset, val_dataset, test_dataset


def get_class_weights(dataset: ReDSM5Dataset) -> torch.Tensor:
    """
    Compute class weights for imbalanced dataset.

    Args:
        dataset: ReDSM5Dataset instance

    Returns:
        Tensor of class weights for use in CrossEntropyLoss
    """
    symptom_indices = [dataset[i]['symptom_idx'].item() for i in range(len(dataset))]
    counts = np.bincount(symptom_indices, minlength=NUM_CLASSES)

    # Avoid division by zero
    counts = np.where(counts == 0, 1, counts)

    # Inverse frequency weighting
    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_CLASSES  # Normalize

    return torch.FloatTensor(weights)


def get_symptom_labels() -> List[str]:
    """Return list of symptom label names."""
    return SYMPTOM_LABELS.copy()


def get_symptom_distribution(dataset: ReDSM5Dataset) -> pd.DataFrame:
    """
    Get distribution of symptoms in dataset.

    Args:
        dataset: ReDSM5Dataset instance

    Returns:
        DataFrame with symptom counts and percentages
    """
    symptom_indices = [dataset[i]['symptom_idx'].item() for i in range(len(dataset))]
    counts = np.bincount(symptom_indices, minlength=NUM_CLASSES)

    df = pd.DataFrame({
        'symptom': SYMPTOM_LABELS,
        'count': counts,
        'percentage': 100 * counts / len(dataset),
    })

    return df
