"""
Binary Classification Dataset for Post-Criterion Matching.

Converts multi-class classification to binary by pairing each post with all criteria.
Each sample is (post_text, criterion_description) → binary label (match/no-match).
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import numpy as np


# DSM-5 Depression Symptom Criteria Descriptions
CRITERION_DESCRIPTIONS = {
    'DEPRESSED_MOOD': 'Depressed mood most of the day, nearly every day (e.g., feels sad, empty, hopeless)',
    'ANHEDONIA': 'Markedly diminished interest or pleasure in all or almost all activities',
    'APPETITE_CHANGE': 'Significant weight loss or gain, or decrease or increase in appetite',
    'SLEEP_ISSUES': 'Insomnia or hypersomnia nearly every day',
    'PSYCHOMOTOR': 'Psychomotor agitation or retardation nearly every day',
    'FATIGUE': 'Fatigue or loss of energy nearly every day',
    'WORTHLESSNESS': 'Feelings of worthlessness or excessive guilt nearly every day',
    'COGNITIVE_ISSUES': 'Diminished ability to think or concentrate, or indecisiveness',
    'SUICIDAL_THOUGHTS': 'Recurrent thoughts of death or suicidal ideation',
    'SPECIAL_CASE': 'Special case or expert discrimination required',
}

CRITERION_LABELS = list(CRITERION_DESCRIPTIONS.keys())
NUM_CRITERIA = len(CRITERION_LABELS)


class BinaryReDSM5Dataset(Dataset):
    """
    Binary classification dataset for post-criterion matching.

    Creates pairs of (post, criterion) with binary labels:
    - Positive (1): Post matches the criterion
    - Negative (0): Post does not match the criterion

    Args:
        posts: List of post texts
        symptom_indices: List of ground truth symptom indices (0-9)
        tokenizer: Huggingface tokenizer
        max_length: Maximum sequence length
        negative_sampling: Strategy for negative examples
            - 'all': Pair each post with all 10 criteria (balanced: 1 pos, 9 neg per post)
            - 'random': Random subset of negative criteria
            - 'hard': Hard negative mining (most similar criteria)
        num_negatives: Number of negative samples per positive (for 'random' sampling)
    """

    def __init__(
        self,
        posts: List[str],
        symptom_indices: List[int],
        tokenizer,
        max_length: int = 512,
        negative_sampling: str = 'all',
        num_negatives: int = 3,
    ):
        self.posts = posts
        self.symptom_indices = symptom_indices
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.negative_sampling = negative_sampling
        self.num_negatives = num_negatives

        # Create (post, criterion, label) pairs
        self.samples = self._create_pairs()

    def _create_pairs(self) -> List[Tuple[str, str, int, int]]:
        """
        Create (post_text, criterion_text, label, post_idx) tuples.

        Returns:
            List of (post, criterion, label, post_idx)
            - post: post text
            - criterion: criterion description
            - label: 1 if match, 0 if no match
            - post_idx: original post index (for tracking)
        """
        samples = []

        for post_idx, (post, true_symptom_idx) in enumerate(zip(self.posts, self.symptom_indices)):
            if self.negative_sampling == 'all':
                # Create pairs with all 10 criteria
                for criterion_idx in range(NUM_CRITERIA):
                    criterion_name = CRITERION_LABELS[criterion_idx]
                    criterion_desc = CRITERION_DESCRIPTIONS[criterion_name]
                    label = 1 if criterion_idx == true_symptom_idx else 0

                    samples.append((post, criterion_desc, label, post_idx))

            elif self.negative_sampling == 'random':
                # Positive sample
                true_criterion_name = CRITERION_LABELS[true_symptom_idx]
                true_criterion_desc = CRITERION_DESCRIPTIONS[true_criterion_name]
                samples.append((post, true_criterion_desc, 1, post_idx))

                # Random negative samples
                negative_indices = [i for i in range(NUM_CRITERIA) if i != true_symptom_idx]
                sampled_negatives = np.random.choice(
                    negative_indices,
                    size=min(self.num_negatives, len(negative_indices)),
                    replace=False
                )

                for neg_idx in sampled_negatives:
                    neg_criterion_name = CRITERION_LABELS[neg_idx]
                    neg_criterion_desc = CRITERION_DESCRIPTIONS[neg_criterion_name]
                    samples.append((post, neg_criterion_desc, 0, post_idx))

            else:
                raise ValueError(f"Unknown negative_sampling: {self.negative_sampling}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        post, criterion, label, post_idx = self.samples[idx]

        # Tokenize as sentence pair: [CLS] post [SEP] criterion [SEP]
        encoding = self.tokenizer(
            post,
            criterion,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'post_idx': torch.tensor(post_idx, dtype=torch.long),
        }

    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of positive/negative samples."""
        labels = [sample[2] for sample in self.samples]
        return {
            'positive': sum(labels),
            'negative': len(labels) - sum(labels),
            'total': len(labels),
            'balance': sum(labels) / len(labels)
        }


def load_redsm5_binary(
    data_dir: Path,
    tokenizer,
    max_length: int = 512,
    negative_sampling: str = 'all',
    num_negatives: int = 3,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42
) -> Tuple[BinaryReDSM5Dataset, BinaryReDSM5Dataset, BinaryReDSM5Dataset]:
    """
    Load ReDSM5 dataset in binary classification format.

    Args:
        data_dir: Path to data directory
        tokenizer: Huggingface tokenizer
        max_length: Max sequence length
        negative_sampling: Negative sampling strategy
        num_negatives: Number of negatives per positive (for 'random')
        test_size: Test split fraction
        val_size: Validation split fraction
        random_seed: Random seed

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    from sklearn.model_selection import train_test_split

    # Load data
    posts_path = data_dir / 'redsm5_posts.csv'
    annotations_path = data_dir / 'redsm5_annotations.csv'

    if not posts_path.exists() or not annotations_path.exists():
        raise FileNotFoundError(f"Data files not found in {data_dir}")

    posts_df = pd.read_csv(posts_path)
    annotations_df = pd.read_csv(annotations_path)

    # Use sentence-level or post-level (depending on what's available)
    if 'text' in annotations_df.columns:
        texts = annotations_df['text'].tolist()
    elif 'text' in posts_df.columns:
        texts = posts_df['text'].tolist()
    else:
        raise ValueError("No 'text' column found in data")

    symptom_indices = annotations_df['symptom_idx'].tolist()

    # Split data (stratified by symptom)
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, symptom_indices,
        test_size=test_size,
        stratify=symptom_indices,
        random_state=random_seed
    )

    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        stratify=y_temp,
        random_state=random_seed
    )

    # Create datasets
    train_dataset = BinaryReDSM5Dataset(
        X_train, y_train, tokenizer, max_length, negative_sampling, num_negatives
    )
    val_dataset = BinaryReDSM5Dataset(
        X_val, y_val, tokenizer, max_length, negative_sampling, num_negatives
    )
    test_dataset = BinaryReDSM5Dataset(
        X_test, y_test, tokenizer, max_length, negative_sampling, num_negatives
    )

    # Print statistics
    print(f"Binary ReDSM5 Dataset Loaded:")
    print(f"  Train: {len(train_dataset)} pairs - {train_dataset.get_class_distribution()}")
    print(f"  Val:   {len(val_dataset)} pairs - {val_dataset.get_class_distribution()}")
    print(f"  Test:  {len(test_dataset)} pairs - {test_dataset.get_class_distribution()}")

    return train_dataset, val_dataset, test_dataset


def get_binary_class_weights(dataset: BinaryReDSM5Dataset) -> torch.Tensor:
    """
    Compute class weights for imbalanced binary classification.

    Args:
        dataset: Binary dataset

    Returns:
        Tensor of shape (2,) with weights for [negative, positive]
    """
    dist = dataset.get_class_distribution()
    total = dist['total']
    n_positive = dist['positive']
    n_negative = dist['negative']

    # Inverse frequency weighting
    weight_positive = total / (2 * n_positive) if n_positive > 0 else 1.0
    weight_negative = total / (2 * n_negative) if n_negative > 0 else 1.0

    return torch.tensor([weight_negative, weight_positive], dtype=torch.float32)
