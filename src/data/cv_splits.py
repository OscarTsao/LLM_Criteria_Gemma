"""
Cross-validation split utilities for ReDSM5 dataset.

Provides functions for creating stratified K-fold splits and loading fold data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

from .redsm5_dataset import ReDSM5Dataset, SYMPTOM_LABELS, NUM_CLASSES


def create_cv_splits(
    annotations_path: str,
    num_folds: int = 5,
    random_seed: int = 42,
    output_dir: Optional[str] = None,
) -> List[Dict[str, np.ndarray]]:
    """
    Create stratified K-fold cross-validation splits.

    Args:
        annotations_path: Path to redsm5_annotations.csv
        num_folds: Number of folds for cross-validation
        random_seed: Random seed for reproducibility
        output_dir: Optional directory to save fold indices

    Returns:
        List of dictionaries, each containing 'train' and 'val' indices
    """
    # Load annotations
    annotations_df = pd.read_csv(annotations_path)

    # Convert symptom labels to indices
    symptom_to_idx = {label: idx for idx, label in enumerate(SYMPTOM_LABELS)}
    annotations_df['symptom_idx'] = annotations_df['symptom_label'].map(symptom_to_idx)

    # Handle unmapped symptoms
    if annotations_df['symptom_idx'].isna().any():
        print(f"Warning: Dropping {annotations_df['symptom_idx'].isna().sum()} unmapped symptoms")
        annotations_df = annotations_df.dropna(subset=['symptom_idx'])

    annotations_df['symptom_idx'] = annotations_df['symptom_idx'].astype(int)

    # Create stratified K-fold splitter
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_seed)

    # Generate splits
    splits = []
    for fold_idx, (train_indices, val_indices) in enumerate(
        skf.split(np.zeros(len(annotations_df)), annotations_df['symptom_idx'])
    ):
        split = {
            'train': train_indices,
            'val': val_indices,
        }
        splits.append(split)

        # Save fold indices if output directory provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save train indices
            train_df = annotations_df.iloc[train_indices].copy()
            train_df.to_csv(output_path / f'fold_{fold_idx}_train.csv', index=False)

            # Save val indices
            val_df = annotations_df.iloc[val_indices].copy()
            val_df.to_csv(output_path / f'fold_{fold_idx}_val.csv', index=False)

            # Save fold metadata
            fold_metadata = {
                'fold': fold_idx,
                'train_size': len(train_indices),
                'val_size': len(val_indices),
                'train_symptom_distribution': train_df['symptom_label'].value_counts().to_dict(),
                'val_symptom_distribution': val_df['symptom_label'].value_counts().to_dict(),
            }

            with open(output_path / f'fold_{fold_idx}_metadata.json', 'w') as f:
                json.dump(fold_metadata, f, indent=2)

    print(f"Created {num_folds} stratified folds")

    # Save overall split metadata
    if output_dir:
        split_metadata = {
            'num_folds': num_folds,
            'random_seed': random_seed,
            'total_samples': len(annotations_df),
            'num_classes': NUM_CLASSES,
        }
        with open(Path(output_dir) / 'split_metadata.json', 'w') as f:
            json.dump(split_metadata, f, indent=2)

    return splits


def load_fold_split(
    data_dir: str,
    fold_idx: int,
    posts_path: Optional[str] = None,
    tokenizer=None,
    max_length: int = 512,
) -> Tuple[ReDSM5Dataset, ReDSM5Dataset]:
    """
    Load a specific fold's train and validation datasets.

    Args:
        data_dir: Directory containing fold CSV files
        fold_idx: Index of fold to load (0-indexed)
        posts_path: Optional path to redsm5_posts.csv (if not provided, assumes 'text' column exists)
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    data_path = Path(data_dir)

    # Load fold annotations
    train_df = pd.read_csv(data_path / f'fold_{fold_idx}_train.csv')
    val_df = pd.read_csv(data_path / f'fold_{fold_idx}_val.csv')

    # If posts_path provided, merge with post texts
    if posts_path:
        posts_df = pd.read_csv(posts_path)
        train_df = pd.merge(train_df, posts_df, on='post_id', how='inner')
        val_df = pd.merge(val_df, posts_df, on='post_id', how='inner')

    # Convert symptom labels to indices
    symptom_to_idx = {label: idx for idx, label in enumerate(SYMPTOM_LABELS)}
    train_df['symptom_idx'] = train_df['symptom_label'].map(symptom_to_idx)
    val_df['symptom_idx'] = val_df['symptom_label'].map(symptom_to_idx)

    # Handle unmapped symptoms
    for df, name in [(train_df, 'train'), (val_df, 'val')]:
        if df['symptom_idx'].isna().any():
            print(f"Warning: Dropping {df['symptom_idx'].isna().sum()} unmapped symptoms from {name}")
            df = df.dropna(subset=['symptom_idx'])

    train_df['symptom_idx'] = train_df['symptom_idx'].astype(int)
    val_df['symptom_idx'] = val_df['symptom_idx'].astype(int)

    # Create datasets
    train_dataset = ReDSM5Dataset(
        texts=train_df['text'].tolist(),
        symptom_indices=train_df['symptom_idx'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )

    val_dataset = ReDSM5Dataset(
        texts=val_df['text'].tolist(),
        symptom_indices=val_df['symptom_idx'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )

    return train_dataset, val_dataset


def get_fold_statistics(splits: List[Dict[str, np.ndarray]]) -> pd.DataFrame:
    """
    Compute statistics for all folds.

    Args:
        splits: List of fold dictionaries with 'train' and 'val' indices

    Returns:
        DataFrame with fold statistics
    """
    stats = []
    for fold_idx, split in enumerate(splits):
        stats.append({
            'fold': fold_idx,
            'train_size': len(split['train']),
            'val_size': len(split['val']),
            'total_size': len(split['train']) + len(split['val']),
            'train_ratio': len(split['train']) / (len(split['train']) + len(split['val'])),
            'val_ratio': len(split['val']) / (len(split['train']) + len(split['val'])),
        })

    return pd.DataFrame(stats)


def load_fold_metadata(data_dir: str, fold_idx: int) -> Dict:
    """
    Load metadata for a specific fold.

    Args:
        data_dir: Directory containing fold files
        fold_idx: Index of fold

    Returns:
        Dictionary containing fold metadata
    """
    metadata_path = Path(data_dir) / f'fold_{fold_idx}_metadata.json'

    if not metadata_path.exists():
        return {}

    with open(metadata_path, 'r') as f:
        return json.load(f)


def verify_fold_stratification(data_dir: str, num_folds: int) -> pd.DataFrame:
    """
    Verify that folds are properly stratified.

    Args:
        data_dir: Directory containing fold files
        num_folds: Number of folds to verify

    Returns:
        DataFrame showing symptom distribution across folds
    """
    results = []

    for fold_idx in range(num_folds):
        metadata = load_fold_metadata(data_dir, fold_idx)

        if not metadata:
            continue

        for split_type in ['train', 'val']:
            dist_key = f'{split_type}_symptom_distribution'
            if dist_key in metadata:
                for symptom, count in metadata[dist_key].items():
                    results.append({
                        'fold': fold_idx,
                        'split': split_type,
                        'symptom': symptom,
                        'count': count,
                    })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Pivot for easier viewing
    pivot_df = df.pivot_table(
        index=['fold', 'split'],
        columns='symptom',
        values='count',
        fill_value=0,
    )

    return pivot_df
