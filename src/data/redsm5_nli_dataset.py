"""
ReDSM5 NLI Dataset for Binary Criteria Matching

Implements NLI-style text-pair classification:
- Input: [CLS] post [SEP] criterion [SEP]
- Output: Binary (0=unmatched, 1=matched)

Generates positive and negative pairs from the original ReDSM5 annotations.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple, List, Dict
from pathlib import Path
import numpy as np
import random

from .dsm5_criteria import get_criterion_text, DSM5_CRITERIA


# Binary labels for NLI-style matching
NUM_CLASSES = 2
LABEL_NAMES = ['unmatched', 'matched']

# Original symptom labels (9 core DSM-5 criteria)
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
]


class ReDSM5NLIDataset(Dataset):
    """
    NLI-style dataset for binary criteria matching.

    Each sample is a (post, criterion) pair with binary label:
    - 1 (matched): post exhibits the criterion
    - 0 (unmatched): post does not exhibit the criterion
    """

    def __init__(
        self,
        post_texts: List[str],
        criterion_texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 512,
    ):
        """
        Initialize NLI dataset.

        Args:
            post_texts: List of Reddit post texts (premises)
            criterion_texts: List of DSM-5 criterion descriptions (hypotheses)
            labels: List of binary labels (0=unmatched, 1=matched)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
        """
        assert len(post_texts) == len(criterion_texts) == len(labels), \
            "All inputs must have same length"

        self.post_texts = post_texts
        self.criterion_texts = criterion_texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        post = str(self.post_texts[idx])
        criterion = str(self.criterion_texts[idx])
        label = self.labels[idx]

        # Format as NSP/NLI: [CLS] post [SEP] criterion [SEP]
        # Using tokenizer's built-in text-pair encoding
        encoding = self.tokenizer(
            post,
            criterion,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
        }


def create_nli_pairs(
    posts_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    negative_ratio: float = 1.0,
    use_short_criteria: bool = False,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Create NLI-style (post, criterion, label) pairs from ALL posts and annotations.

    Uses all posts from posts.csv paired with all 9 criteria:
    - Positive pairs (label=1): annotations with status=1
    - Negative pairs (label=0): annotations with status=0 OR posts not annotated for that criterion

    Args:
        posts_df: DataFrame with columns [post_id, text]
        annotations_df: DataFrame with columns [post_id, symptom_label, status]
        negative_ratio: Ratio of negative to positive samples (for additional sampling)
        use_short_criteria: Whether to use short criterion descriptions
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with columns [post_id, post_text, criterion_text, label, symptom_label]
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    pairs = []

    # Create annotation lookup: {(post_id, symptom): status}
    annotation_lookup = {}
    for _, row in annotations_df.iterrows():
        key = (row['post_id'], row['symptom_label'])
        annotation_lookup[key] = row['status']

    # Process ALL posts with ALL 9 criteria
    for _, post_row in posts_df.iterrows():
        post_id = post_row['post_id']
        post_text = post_row['text']

        # Pair this post with each of the 9 criteria
        for symptom_label in SYMPTOM_LABELS:
            criterion_text = get_criterion_text(symptom_label, use_short=use_short_criteria)

            # Check annotation status
            key = (post_id, symptom_label)
            status = annotation_lookup.get(key, None)

            # Determine label based on status
            if status == 1:
                # Positive pair: post exhibits this criterion
                label = 1
                pair_type = 'positive_status1'
            elif status == 0:
                # Negative pair: explicitly marked as not exhibiting
                label = 0
                pair_type = 'negative_status0'
            else:
                # Not annotated for this criterion - treat as negative
                label = 0
                pair_type = 'negative_not_annotated'

            pairs.append({
                'post_id': post_id,
                'post_text': post_text,
                'criterion_text': criterion_text,
                'label': label,
                'symptom_label': symptom_label,
                'pair_type': pair_type,
                'status': status if status is not None else -1,
            })

    pairs_df = pd.DataFrame(pairs)

    # Count different types
    num_positives = len(pairs_df[pairs_df['label'] == 1])
    num_negatives_status0 = len(pairs_df[pairs_df['pair_type'] == 'negative_status0'])
    num_negatives_not_annotated = len(pairs_df[pairs_df['pair_type'] == 'negative_not_annotated'])
    num_total_negatives = len(pairs_df[pairs_df['label'] == 0])

    # Shuffle pairs
    pairs_df = pairs_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    num_posts = len(posts_df)
    num_criteria = len(SYMPTOM_LABELS)

    print(f"Created {len(pairs_df)} NLI pairs from {num_posts} posts × {num_criteria} criteria:")
    print(f"  Positive (status=1): {num_positives}")
    print(f"  Negative (status=0): {num_negatives_status0}")
    print(f"  Negative (not annotated): {num_negatives_not_annotated}")
    print(f"  Total negatives: {num_total_negatives}")
    print(f"  Total pairs: {len(pairs_df)}")
    print(f"  Balance: {num_total_negatives}/{num_positives} = {num_total_negatives/num_positives:.2f}:1")

    return pairs_df


def load_redsm5_nli(
    data_dir: str,
    tokenizer,
    max_length: int = 512,
    test_size: float = 0.15,
    val_size: float = 0.15,
    negative_ratio: float = 1.0,
    use_short_criteria: bool = False,
    random_seed: int = 42,
) -> Tuple[ReDSM5NLIDataset, ReDSM5NLIDataset, ReDSM5NLIDataset, pd.DataFrame]:
    """
    Load ReDSM5 dataset as NLI-style binary classification task.

    Args:
        data_dir: Path to directory containing CSV files
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        test_size: Proportion for test set
        val_size: Proportion for validation set
        negative_ratio: Ratio of negative to positive samples
        use_short_criteria: Whether to use short criterion descriptions
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, pairs_df)
    """
    data_path = Path(data_dir)

    # Load posts and annotations
    posts_df = pd.read_csv(data_path / 'redsm5_posts.csv')
    annotations_df = pd.read_csv(data_path / 'redsm5_annotations.csv')

    # Create NLI pairs
    pairs_df = create_nli_pairs(
        posts_df,
        annotations_df,
        negative_ratio=negative_ratio,
        use_short_criteria=use_short_criteria,
        random_seed=random_seed,
    )

    # Extract data
    post_texts = pairs_df['post_text'].tolist()
    criterion_texts = pairs_df['criterion_text'].tolist()
    labels = pairs_df['label'].tolist()

    # Stratified train/test split
    X_posts_temp, X_posts_test, X_criteria_temp, X_criteria_test, y_temp, y_test = train_test_split(
        post_texts,
        criterion_texts,
        labels,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels,
    )

    # Stratified train/val split
    val_ratio = val_size / (1 - test_size)
    X_posts_train, X_posts_val, X_criteria_train, X_criteria_val, y_train, y_val = train_test_split(
        X_posts_temp,
        X_criteria_temp,
        y_temp,
        test_size=val_ratio,
        random_state=random_seed,
        stratify=y_temp,
    )

    # Create datasets
    train_dataset = ReDSM5NLIDataset(
        X_posts_train, X_criteria_train, y_train, tokenizer, max_length
    )
    val_dataset = ReDSM5NLIDataset(
        X_posts_val, X_criteria_val, y_val, tokenizer, max_length
    )
    test_dataset = ReDSM5NLIDataset(
        X_posts_test, X_criteria_test, y_test, tokenizer, max_length
    )

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    # Print label distribution
    for name, dataset in [('Train', train_dataset), ('Val', val_dataset), ('Test', test_dataset)]:
        matched = sum(dataset.labels)
        unmatched = len(dataset.labels) - matched
        print(f"  {name} balance: {matched} matched, {unmatched} unmatched "
              f"({matched / len(dataset.labels) * 100:.1f}% positive)")

    return train_dataset, val_dataset, test_dataset, pairs_df


def get_class_weights(dataset: ReDSM5NLIDataset) -> torch.Tensor:
    """
    Compute class weights for binary classification.

    Args:
        dataset: ReDSM5NLIDataset instance

    Returns:
        Tensor of class weights [weight_unmatched, weight_matched]
    """
    labels = dataset.labels
    counts = np.bincount(labels, minlength=NUM_CLASSES)

    # Avoid division by zero
    counts = np.where(counts == 0, 1, counts)

    # Inverse frequency weighting
    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_CLASSES  # Normalize

    return torch.FloatTensor(weights)
