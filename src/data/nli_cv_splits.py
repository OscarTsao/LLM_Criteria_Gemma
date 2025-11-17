"""
Cross-Validation Splits for NLI-style Criteria Matching

Creates stratified K-fold splits for binary NLI task.
Ensures balanced positive/negative distribution across folds.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from typing import Tuple, List, Dict
import json
from torch.utils.data import Dataset

from .redsm5_nli_dataset import ReDSM5NLIDataset, create_nli_pairs


def create_nli_cv_splits(
    data_dir: str,
    output_dir: str,
    num_folds: int = 5,
    negative_ratio: float = 1.0,
    use_short_criteria: bool = False,
    random_seed: int = 42,
) -> Dict:
    """
    Create stratified K-fold CV splits for NLI pairs.

    Args:
        data_dir: Path to ReDSM5 data directory
        output_dir: Path to save fold splits
        num_folds: Number of CV folds
        negative_ratio: Ratio of negative to positive samples
        use_short_criteria: Whether to use short criterion descriptions
        random_seed: Random seed

    Returns:
        Dictionary with fold metadata
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Creating {num_folds}-fold CV splits for NLI task...")

    # Load data
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

    # Save full pairs dataset
    pairs_path = output_path / 'nli_pairs_full.csv'
    pairs_df.to_csv(pairs_path, index=False)
    print(f"Saved full NLI pairs to: {pairs_path}")

    # Create stratified K-fold splits
    skf = StratifiedKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=random_seed
    )

    # Compute overall statistics
    num_positive = int(pairs_df['label'].sum())
    num_negative = int((pairs_df['label'] == 0).sum())
    num_status0 = len(pairs_df[pairs_df['pair_type'] == 'negative_status0'])
    num_not_annotated = len(pairs_df[pairs_df['pair_type'] == 'negative_not_annotated'])
    num_posts = len(posts_df)
    num_criteria = len(pairs_df['symptom_label'].unique())

    fold_info = {
        'num_folds': num_folds,
        'total_samples': len(pairs_df),
        'num_posts': num_posts,
        'num_criteria': num_criteria,
        'num_positive': num_positive,
        'num_negative': num_negative,
        'num_negative_status0': num_status0,
        'num_negative_not_annotated': num_not_annotated,
        'negative_ratio': negative_ratio,
        'use_short_criteria': use_short_criteria,
        'random_seed': random_seed,
        'folds': []
    }

    # Generate folds
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(pairs_df, pairs_df['label'])):
        fold_num = fold_idx + 1

        # Split data
        train_df = pairs_df.iloc[train_idx].reset_index(drop=True)
        val_df = pairs_df.iloc[val_idx].reset_index(drop=True)

        # Save fold splits
        train_path = output_path / f'fold_{fold_num}_train.csv'
        val_path = output_path / f'fold_{fold_num}_val.csv'

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        # Compute fold statistics
        train_pos = int(train_df['label'].sum())
        train_neg = len(train_df) - train_pos
        val_pos = int(val_df['label'].sum())
        val_neg = len(val_df) - val_pos

        fold_stat = {
            'fold': fold_num,
            'train_size': len(train_df),
            'val_size': len(val_df),
            'train_positive': train_pos,
            'train_negative': train_neg,
            'val_positive': val_pos,
            'val_negative': val_neg,
            'train_positive_ratio': train_pos / len(train_df),
            'val_positive_ratio': val_pos / len(val_df),
        }

        fold_info['folds'].append(fold_stat)

        print(f"\nFold {fold_num}:")
        print(f"  Train: {len(train_df)} samples ({train_pos} pos, {train_neg} neg, "
              f"{train_pos / len(train_df) * 100:.1f}% pos)")
        print(f"  Val: {len(val_df)} samples ({val_pos} pos, {val_neg} neg, "
              f"{val_pos / len(val_df) * 100:.1f}% pos)")

    # Save fold metadata
    metadata_path = output_path / 'nli_cv_folds_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(fold_info, f, indent=2)

    print(f"\nFold metadata saved to: {metadata_path}")
    print(f"All fold splits saved to: {output_path}")

    return fold_info


def load_nli_fold_split(
    fold_dir: str,
    fold_num: int,
    tokenizer,
    max_length: int = 512,
) -> Tuple[ReDSM5NLIDataset, ReDSM5NLIDataset]:
    """
    Load a specific fold split for training.

    Args:
        fold_dir: Directory containing fold CSV files
        fold_num: Fold number (1-indexed)
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    fold_path = Path(fold_dir)

    # Load fold data
    train_df = pd.read_csv(fold_path / f'fold_{fold_num}_train.csv')
    val_df = pd.read_csv(fold_path / f'fold_{fold_num}_val.csv')

    # Extract data
    train_posts = train_df['post_text'].tolist()
    train_criteria = train_df['criterion_text'].tolist()
    train_labels = train_df['label'].tolist()

    val_posts = val_df['post_text'].tolist()
    val_criteria = val_df['criterion_text'].tolist()
    val_labels = val_df['label'].tolist()

    # Create datasets
    train_dataset = ReDSM5NLIDataset(
        train_posts, train_criteria, train_labels, tokenizer, max_length
    )
    val_dataset = ReDSM5NLIDataset(
        val_posts, val_criteria, val_labels, tokenizer, max_length
    )

    print(f"Loaded Fold {fold_num}:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")

    return train_dataset, val_dataset


def get_nli_fold_metadata(fold_dir: str) -> Dict:
    """
    Load fold metadata.

    Args:
        fold_dir: Directory containing fold metadata

    Returns:
        Dictionary with fold information
    """
    metadata_path = Path(fold_dir) / 'nli_cv_folds_metadata.json'

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return metadata
