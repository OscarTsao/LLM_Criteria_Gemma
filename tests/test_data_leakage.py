"""
Data leakage and cross-validation integrity tests.

Ensures that:
1. No post_id appears in both train and validation for any fold
2. All folds combined cover the entire dataset exactly once
3. Stratification is properly maintained
4. Sentence counts match expectations
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.cv_splits import create_cv_splits, load_fold_split, get_fold_statistics


class TestDataLeakage:
    """Test suite for data leakage prevention."""

    @pytest.fixture
    def mock_annotations(self, tmp_path):
        """Create mock annotations for testing."""
        # Create 100 mock samples with 5 classes
        np.random.seed(42)
        n_samples = 100

        data = {
            'post_id': [f"post_{i}" for i in range(n_samples)],
            'sentence_id': [f"post_{i}_sent_0" for i in range(n_samples)],
            'text': [f"Sample text {i}" for i in range(n_samples)],
            'symptom_idx': np.random.randint(0, 5, size=n_samples)
        }

        df = pd.DataFrame(data)
        annotations_path = tmp_path / "annotations.csv"
        df.to_csv(annotations_path, index=False)

        return annotations_path, df

    def test_no_postid_overlap_between_train_val(self, mock_annotations):
        """
        CRITICAL: Ensure no post_id appears in both train and val within a fold.
        """
        annotations_path, df = mock_annotations

        # Create splits
        splits = create_cv_splits(
            annotations_path,
            num_folds=5,
            random_seed=42,
            output_dir=None  # Don't save
        )

        for fold_idx in range(5):
            train_df = splits[fold_idx]['train']
            val_df = splits[fold_idx]['val']

            # Get unique post_ids
            train_post_ids = set(train_df['post_id'].unique())
            val_post_ids = set(val_df['post_id'].unique())

            # Check for overlap
            overlap = train_post_ids.intersection(val_post_ids)

            assert len(overlap) == 0, (
                f"Fold {fold_idx}: Found {len(overlap)} overlapping post_ids "
                f"between train and val! Examples: {list(overlap)[:5]}"
            )

    def test_all_folds_cover_full_dataset(self, mock_annotations):
        """
        Ensure that all validation sets combined cover the entire dataset exactly once.
        """
        annotations_path, df = mock_annotations

        splits = create_cv_splits(
            annotations_path,
            num_folds=5,
            random_seed=42,
            output_dir=None
        )

        # Collect all validation post_ids across folds
        all_val_post_ids = []
        for fold_idx in range(5):
            val_df = splits[fold_idx]['val']
            all_val_post_ids.extend(val_df['post_id'].unique())

        # Check that we have exactly the right number (each post once)
        original_post_ids = set(df['post_id'].unique())
        val_post_ids_set = set(all_val_post_ids)

        assert len(all_val_post_ids) == len(original_post_ids), (
            f"Expected {len(original_post_ids)} unique posts across all val sets, "
            f"but got {len(all_val_post_ids)}"
        )

        # Check no duplicates
        assert len(all_val_post_ids) == len(val_post_ids_set), (
            "Some posts appear in multiple validation sets!"
        )

        # Check all posts are covered
        missing = original_post_ids - val_post_ids_set
        extra = val_post_ids_set - original_post_ids

        assert len(missing) == 0, f"Missing posts: {missing}"
        assert len(extra) == 0, f"Extra posts: {extra}"

    def test_stratification_maintained(self, mock_annotations):
        """
        Ensure class distribution is approximately balanced across folds.
        """
        annotations_path, df = mock_annotations

        splits = create_cv_splits(
            annotations_path,
            num_folds=5,
            random_seed=42,
            output_dir=None
        )

        # Get global class distribution
        global_dist = df['symptom_idx'].value_counts(normalize=True).sort_index()

        for fold_idx in range(5):
            val_df = splits[fold_idx]['val']

            # Get fold validation distribution
            fold_dist = val_df['symptom_idx'].value_counts(normalize=True).sort_index()

            # Check that distributions are similar (within 15% tolerance)
            for class_idx in global_dist.index:
                global_prop = global_dist[class_idx]
                fold_prop = fold_dist.get(class_idx, 0)

                # Allow some tolerance due to small sample size
                tolerance = 0.15
                assert abs(fold_prop - global_prop) <= tolerance, (
                    f"Fold {fold_idx}, Class {class_idx}: "
                    f"Distribution mismatch (global={global_prop:.3f}, fold={fold_prop:.3f})"
                )

    def test_sentence_counts_per_post(self, tmp_path):
        """
        Validate that sentence counts per post are handled correctly.
        """
        # Create mock data with multiple sentences per post
        data = {
            'post_id': ['post_0', 'post_0', 'post_0', 'post_1', 'post_1', 'post_2'],
            'sentence_id': ['post_0_sent_0', 'post_0_sent_1', 'post_0_sent_2',
                           'post_1_sent_0', 'post_1_sent_1', 'post_2_sent_0'],
            'text': ['Text A', 'Text B', 'Text C', 'Text D', 'Text E', 'Text F'],
            'symptom_idx': [0, 0, 0, 1, 1, 2]
        }

        df = pd.DataFrame(data)
        annotations_path = tmp_path / "annotations_multi_sent.csv"
        df.to_csv(annotations_path, index=False)

        splits = create_cv_splits(
            annotations_path,
            num_folds=3,
            random_seed=42,
            output_dir=None
        )

        # Verify that all sentences from a post stay together
        for fold_idx in range(3):
            val_df = splits[fold_idx]['val']

            # For each post in val, check all its sentences are present
            for post_id in val_df['post_id'].unique():
                original_sentences = df[df['post_id'] == post_id]['sentence_id'].values
                val_sentences = val_df[val_df['post_id'] == post_id]['sentence_id'].values

                assert set(original_sentences) == set(val_sentences), (
                    f"Fold {fold_idx}, Post {post_id}: "
                    f"Missing sentences in validation set"
                )

    def test_very_long_post_handling(self, tmp_path):
        """
        Test that very long posts (>100 sentences) are handled correctly.
        """
        # Create a post with 150 sentences
        n_sentences = 150
        data = {
            'post_id': [f"long_post_0"] * n_sentences,
            'sentence_id': [f"long_post_0_sent_{i}" for i in range(n_sentences)],
            'text': [f"Sentence {i}" for i in range(n_sentences)],
            'symptom_idx': [0] * n_sentences  # All same class
        }

        df = pd.DataFrame(data)
        annotations_path = tmp_path / "long_post.csv"
        df.to_csv(annotations_path, index=False)

        splits = create_cv_splits(
            annotations_path,
            num_folds=5,
            random_seed=42,
            output_dir=None
        )

        # The long post should appear in exactly one validation fold
        folds_with_post = 0
        for fold_idx in range(5):
            val_df = splits[fold_idx]['val']
            if 'long_post_0' in val_df['post_id'].values:
                folds_with_post += 1

                # All 150 sentences should be present
                assert len(val_df) == n_sentences, (
                    f"Long post should have all {n_sentences} sentences, "
                    f"but got {len(val_df)}"
                )

        assert folds_with_post == 1, (
            f"Long post should appear in exactly 1 fold, but found in {folds_with_post}"
        )

    def test_deterministic_splits(self, mock_annotations):
        """
        Ensure splits are deterministic given the same random seed.
        """
        annotations_path, df = mock_annotations

        splits1 = create_cv_splits(annotations_path, num_folds=5, random_seed=42)
        splits2 = create_cv_splits(annotations_path, num_folds=5, random_seed=42)

        # Check that splits are identical
        for fold_idx in range(5):
            train1 = splits1[fold_idx]['train']
            train2 = splits2[fold_idx]['train']

            val1 = splits1[fold_idx]['val']
            val2 = splits2[fold_idx]['val']

            # Compare post_ids
            assert set(train1['post_id']) == set(train2['post_id']), (
                f"Fold {fold_idx}: Non-deterministic train split"
            )
            assert set(val1['post_id']) == set(val2['post_id']), (
                f"Fold {fold_idx}: Non-deterministic val split"
            )

    def test_empty_fold_handling(self, tmp_path):
        """
        Test behavior with very small datasets (edge case).
        """
        # Create tiny dataset (10 samples, 5 folds -> 2 per fold)
        data = {
            'post_id': [f"post_{i}" for i in range(10)],
            'sentence_id': [f"post_{i}_sent_0" for i in range(10)],
            'text': [f"Text {i}" for i in range(10)],
            'symptom_idx': [i % 2 for i in range(10)]  # 2 classes
        }

        df = pd.DataFrame(data)
        annotations_path = tmp_path / "tiny.csv"
        df.to_csv(annotations_path, index=False)

        splits = create_cv_splits(
            annotations_path,
            num_folds=5,
            random_seed=42,
            output_dir=None
        )

        # Each fold should have some validation data
        for fold_idx in range(5):
            val_df = splits[fold_idx]['val']
            assert len(val_df) > 0, f"Fold {fold_idx} has empty validation set"


class TestFoldStatistics:
    """Test fold statistics utilities."""

    @pytest.fixture
    def sample_splits(self):
        """Create sample splits for testing."""
        splits = {}
        for fold_idx in range(3):
            train_df = pd.DataFrame({
                'post_id': [f"train_{fold_idx}_{i}" for i in range(100)],
                'symptom_idx': np.random.randint(0, 5, 100)
            })
            val_df = pd.DataFrame({
                'post_id': [f"val_{fold_idx}_{i}" for i in range(25)],
                'symptom_idx': np.random.randint(0, 5, 25)
            })
            splits[fold_idx] = {'train': train_df, 'val': val_df}

        return splits

    def test_get_fold_statistics(self, sample_splits):
        """Test fold statistics computation."""
        stats = get_fold_statistics(sample_splits)

        assert isinstance(stats, pd.DataFrame)
        assert len(stats) == 3  # 3 folds
        assert 'fold' in stats.columns
        assert 'train_size' in stats.columns
        assert 'val_size' in stats.columns
        assert all(stats['train_size'] == 100)
        assert all(stats['val_size'] == 25)
