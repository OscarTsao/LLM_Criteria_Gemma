"""
ReDSM5 NLI Dataset Loader for Binary Criteria Matching.

Converts the multi-class ReDSM5 task into a binary NLI task where each example
is a (post, criterion) pair labeled as matched (1) or unmatched (0).

For 1,484 posts × 9 criteria = 13,356 total examples.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from typing import List, Optional, Tuple, Dict

from .annotations_utils import ensure_symptom_label_column
from .dsm5_criteria import get_criterion_text, get_symptom_labels, NUM_CRITERIA


# Binary classification: 0 = unmatched, 1 = matched
NUM_CLASSES = 2


class ReDSM5NLIDataset(Dataset):
    """PyTorch Dataset for ReDSM5 Binary Criteria Matching (NLI-style)."""

    def __init__(
        self,
        post_texts: List[str],
        criterion_texts: List[str],
        labels: List[int],
        post_ids: List[str],
        criterion_names: List[str],
        tokenizer,
        max_length: int = 512,
    ):
        """
        Initialize ReDSM5 NLI Dataset.

        Args:
            post_texts: List of post text strings
            criterion_texts: List of criterion description strings
            labels: List of binary labels (0=unmatched, 1=matched)
            post_ids: List of post IDs for tracking
            criterion_names: List of criterion names (e.g., 'DEPRESSED_MOOD')
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.post_texts = post_texts
        self.criterion_texts = criterion_texts
        self.labels = labels
        self.post_ids = post_ids
        self.criterion_names = criterion_names
        self.tokenizer = tokenizer
        self.max_length = max_length

        assert len(post_texts) == len(criterion_texts) == len(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        post_text = str(self.post_texts[idx])
        criterion_text = str(self.criterion_texts[idx])
        label = self.labels[idx]

        # Format: [CLS] post [SEP] criterion [SEP]
        # For models without explicit CLS/SEP tokens, tokenizer handles this automatically
        encoding = self.tokenizer(
            post_text,
            criterion_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'post_id': self.post_ids[idx],
            'criterion_name': self.criterion_names[idx],
        }


def create_post_criterion_pairs(
    posts_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
) -> Tuple[List[str], List[str], List[int], List[str], List[str]]:
    """
    Create post-criterion pairs with binary labels.

    For each post, creates 9 pairs (one for each criterion).
    Label = 1 if the post matches that criterion, 0 otherwise.

    Args:
        posts_df: DataFrame with columns ['post_id', 'text']
        annotations_df: DataFrame with columns ['post_id', 'symptom_label']

    Returns:
        Tuple of (post_texts, criterion_texts, labels, post_ids, criterion_names)
    """
    # Get the 9 DSM-5 symptom labels (excluding SPECIAL_CASE)
    symptom_labels = get_symptom_labels()

    # Create a mapping of post_id -> set of matching symptoms
    post_symptom_map = {}
    for _, row in annotations_df.iterrows():
        post_id = row['post_id']
        symptom = row['symptom_label']

        # Only include the 9 core DSM-5 symptoms (exclude SPECIAL_CASE)
        if symptom in symptom_labels:
            if post_id not in post_symptom_map:
                post_symptom_map[post_id] = set()
            post_symptom_map[post_id].add(symptom)

    # Create pairs
    post_texts = []
    criterion_texts = []
    labels = []
    post_ids = []
    criterion_names = []

    for _, post_row in posts_df.iterrows():
        post_id = post_row['post_id']
        post_text = post_row['text']

        # Get symptoms that match this post (default to empty set if none)
        matching_symptoms = post_symptom_map.get(post_id, set())

        # Create 9 pairs for this post (one for each criterion)
        for symptom_label in symptom_labels:
            criterion_text = get_criterion_text(symptom_label)

            # Label = 1 if this criterion matches, 0 otherwise
            label = 1 if symptom_label in matching_symptoms else 0

            post_texts.append(post_text)
            criterion_texts.append(criterion_text)
            labels.append(label)
            post_ids.append(post_id)
            criterion_names.append(symptom_label)

    return post_texts, criterion_texts, labels, post_ids, criterion_names


def load_redsm5_nli(
    data_dir: str,
    tokenizer,
    max_length: int = 512,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42,
    post_limit: Optional[int] = None,
) -> Tuple[ReDSM5NLIDataset, ReDSM5NLIDataset, ReDSM5NLIDataset]:
    """
    Load ReDSM5 dataset for binary NLI criteria matching task.

    Creates post-criterion pairs where each example is:
    - Input: [CLS] post [SEP] criterion [SEP]
    - Output: matched (1) or unmatched (0)

    Total examples: 1,484 posts × 9 criteria = 13,356 pairs

    Args:
        data_dir: Path to directory containing redsm5_posts.csv and redsm5_annotations.csv
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        test_size: Proportion for test set (by unique posts)
        val_size: Proportion for validation set (by unique posts)
        random_seed: Random seed for reproducibility
        post_limit: Optional cap on the number of unique posts (useful for debug)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_path = Path(data_dir)

    # Load posts and annotations
    posts_df = pd.read_csv(data_path / 'redsm5_posts.csv')
    annotations_df = pd.read_csv(data_path / 'redsm5_annotations.csv')
    annotations_df = ensure_symptom_label_column(annotations_df)

    print(f"Loaded {len(posts_df)} posts and {len(annotations_df)} annotations")

    # Create post-criterion pairs
    post_texts, criterion_texts, labels, post_ids_list, criterion_names = \
        create_post_criterion_pairs(posts_df, annotations_df)

    print(f"Created {len(labels)} post-criterion pairs")
    print(f"  Matched (label=1): {sum(labels)}")
    print(f"  Unmatched (label=0): {len(labels) - sum(labels)}")

    # Split by unique posts (not by pairs) to avoid data leakage
    # Group pairs by post_id
    unique_post_ids = posts_df['post_id'].unique()

    if post_limit is not None and post_limit > 0:
        if post_limit < len(unique_post_ids):
            rng = np.random.default_rng(random_seed)
            unique_post_ids = rng.choice(unique_post_ids, size=post_limit, replace=False)
        else:
            # If limit larger than dataset, just use all posts
            post_limit = None

    # Split post IDs into train/val/test
    train_post_ids, test_post_ids = train_test_split(
        unique_post_ids,
        test_size=test_size,
        random_state=random_seed,
    )

    if val_size <= 0:
        val_post_ids = np.array([], dtype=train_post_ids.dtype)
    else:
        val_ratio = val_size / (1 - test_size)
        train_post_ids, val_post_ids = train_test_split(
            train_post_ids,
            test_size=val_ratio,
            random_state=random_seed,
        )

    # Convert to sets for fast lookup
    train_post_set = set(train_post_ids)
    val_post_set = set(val_post_ids)
    test_post_set = set(test_post_ids)

    # Split pairs based on which set the post belongs to
    def split_pairs(post_set):
        indices = [i for i, pid in enumerate(post_ids_list) if pid in post_set]
        return (
            [post_texts[i] for i in indices],
            [criterion_texts[i] for i in indices],
            [labels[i] for i in indices],
            [post_ids_list[i] for i in indices],
            [criterion_names[i] for i in indices],
        )

    train_data = split_pairs(train_post_set)
    val_data = split_pairs(val_post_set)
    test_data = split_pairs(test_post_set)

    # Create datasets
    train_dataset = ReDSM5NLIDataset(*train_data, tokenizer, max_length)
    val_dataset = ReDSM5NLIDataset(*val_data, tokenizer, max_length)
    test_dataset = ReDSM5NLIDataset(*test_data, tokenizer, max_length)

    print(f"\nDataset split complete:")
    print(f"  Train: {len(train_dataset)} pairs ({len(train_post_set)} posts)")
    print(f"  Val: {len(val_dataset)} pairs ({len(val_post_set)} posts)")
    print(f"  Test: {len(test_dataset)} pairs ({len(test_post_set)} posts)")

    # Show class balance
    for split_name, dataset in [('Train', train_dataset), ('Val', val_dataset), ('Test', test_dataset)]:
        n_matched = sum(dataset.labels)
        n_total = len(dataset.labels)
        match_pct = 100 * n_matched / n_total if n_total else 0.0
        print(f"  {split_name} balance: {n_matched}/{n_total} matched ({match_pct:.1f}%)")

    return train_dataset, val_dataset, test_dataset


def get_class_weights(dataset: ReDSM5NLIDataset) -> torch.Tensor:
    """
    Compute class weights for imbalanced binary classification.

    Args:
        dataset: ReDSM5NLIDataset instance

    Returns:
        Tensor of class weights [weight_for_0, weight_for_1]
    """
    labels = np.array(dataset.labels)
    counts = np.bincount(labels, minlength=NUM_CLASSES)

    # Avoid division by zero
    counts = np.where(counts == 0, 1, counts)

    # Inverse frequency weighting
    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_CLASSES  # Normalize

    return torch.FloatTensor(weights)


def get_label_distribution(dataset: ReDSM5NLIDataset) -> pd.DataFrame:
    """
    Get distribution of labels in dataset.

    Args:
        dataset: ReDSM5NLIDataset instance

    Returns:
        DataFrame with label counts and percentages
    """
    labels = np.array(dataset.labels)
    counts = np.bincount(labels, minlength=NUM_CLASSES)

    df = pd.DataFrame({
        'label': ['unmatched', 'matched'],
        'count': counts,
        'percentage': 100 * counts / len(dataset),
    })

    return df


def get_criterion_distribution(dataset: ReDSM5NLIDataset) -> pd.DataFrame:
    """
    Get distribution of matches per criterion.

    Args:
        dataset: ReDSM5NLIDataset instance

    Returns:
        DataFrame with per-criterion match statistics
    """
    # Group by criterion name
    criterion_stats = {}
    for i in range(len(dataset)):
        criterion = dataset.criterion_names[i]
        label = dataset.labels[i]

        if criterion not in criterion_stats:
            criterion_stats[criterion] = {'matched': 0, 'unmatched': 0}

        if label == 1:
            criterion_stats[criterion]['matched'] += 1
        else:
            criterion_stats[criterion]['unmatched'] += 1

    # Convert to DataFrame
    rows = []
    for criterion, stats in criterion_stats.items():
        total = stats['matched'] + stats['unmatched']
        rows.append({
            'criterion': criterion,
            'matched': stats['matched'],
            'unmatched': stats['unmatched'],
            'total': total,
            'match_rate': 100 * stats['matched'] / total if total > 0 else 0,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('matched', ascending=False)

    return df
