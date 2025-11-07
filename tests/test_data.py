"""
Unit tests for data loading and processing.

Tests ReDSM5 dataset loading, splitting, and cross-validation utilities.
"""

import pytest
import torch
import pandas as pd
import numpy as np
import tempfile
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.redsm5_dataset import (
    ReDSM5Dataset,
    get_class_weights,
    get_symptom_labels,
    NUM_CLASSES,
    SYMPTOM_LABELS,
)
from data.cv_splits import (
    create_cv_splits,
    get_fold_statistics,
)


class TestConstants:
    """Test dataset constants."""

    def test_num_classes(self):
        """Test NUM_CLASSES is correct."""
        assert NUM_CLASSES == 10

    def test_symptom_labels_length(self):
        """Test SYMPTOM_LABELS has correct length."""
        assert len(SYMPTOM_LABELS) == NUM_CLASSES

    def test_symptom_labels_content(self):
        """Test SYMPTOM_LABELS contains expected symptoms."""
        expected_symptoms = [
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
        assert SYMPTOM_LABELS == expected_symptoms

    def test_get_symptom_labels(self):
        """Test get_symptom_labels returns copy."""
        labels1 = get_symptom_labels()
        labels2 = get_symptom_labels()

        assert labels1 == labels2
        assert labels1 is not labels2  # Should be different objects


class TestReDSM5Dataset:
    """Tests for ReDSM5Dataset class."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.return_value = {
            'input_ids': torch.randint(0, 1000, (1, 512)),
            'attention_mask': torch.ones(1, 512),
        }
        return tokenizer

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        texts = [
            "I feel sad and hopeless every day.",
            "Nothing brings me joy anymore.",
            "I can't sleep at night.",
        ]
        symptom_indices = [0, 1, 3]  # DEPRESSED_MOOD, ANHEDONIA, SLEEP_ISSUES
        return texts, symptom_indices

    def test_dataset_initialization(self, mock_tokenizer, sample_data):
        """Test ReDSM5Dataset initialization."""
        texts, symptom_indices = sample_data
        dataset = ReDSM5Dataset(
            texts=texts,
            symptom_indices=symptom_indices,
            tokenizer=mock_tokenizer,
            max_length=512,
        )

        assert len(dataset) == 3
        assert dataset.texts == texts
        assert dataset.symptom_indices == symptom_indices
        assert dataset.max_length == 512

    def test_dataset_length(self, mock_tokenizer, sample_data):
        """Test dataset __len__ method."""
        texts, symptom_indices = sample_data
        dataset = ReDSM5Dataset(texts, symptom_indices, mock_tokenizer, 512)

        assert len(dataset) == 3

    def test_dataset_getitem(self, mock_tokenizer, sample_data):
        """Test dataset __getitem__ method."""
        texts, symptom_indices = sample_data
        dataset = ReDSM5Dataset(texts, symptom_indices, mock_tokenizer, 512)

        item = dataset[0]

        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'symptom_idx' in item

        assert item['input_ids'].shape == (512,)
        assert item['attention_mask'].shape == (512,)
        assert item['symptom_idx'].item() == 0

    def test_dataset_getitem_all_indices(self, mock_tokenizer, sample_data):
        """Test dataset returns correct symptom indices."""
        texts, symptom_indices = sample_data
        dataset = ReDSM5Dataset(texts, symptom_indices, mock_tokenizer, 512)

        for i in range(len(dataset)):
            item = dataset[i]
            assert item['symptom_idx'].item() == symptom_indices[i]

    def test_dataset_tokenizer_called_correctly(self, sample_data):
        """Test that tokenizer is called with correct parameters."""
        texts, symptom_indices = sample_data
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': torch.randint(0, 1000, (1, 256)),
            'attention_mask': torch.ones(1, 256),
        }

        dataset = ReDSM5Dataset(texts, symptom_indices, mock_tokenizer, max_length=256)
        _ = dataset[0]

        mock_tokenizer.assert_called_once()
        call_args = mock_tokenizer.call_args
        assert call_args.kwargs['max_length'] == 256
        assert call_args.kwargs['padding'] == 'max_length'
        assert call_args.kwargs['truncation'] is True
        assert call_args.kwargs['return_tensors'] == 'pt'

    def test_dataset_handles_string_conversion(self, mock_tokenizer):
        """Test dataset converts non-string texts properly."""
        texts = [123, None, "valid text"]  # Mixed types
        symptom_indices = [0, 1, 2]

        dataset = ReDSM5Dataset(texts, symptom_indices, mock_tokenizer, 512)

        # Should not raise error
        item = dataset[0]
        assert 'input_ids' in item


class TestGetClassWeights:
    """Tests for get_class_weights function."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.return_value = {
            'input_ids': torch.randint(0, 1000, (1, 512)),
            'attention_mask': torch.ones(1, 512),
        }
        return tokenizer

    def test_class_weights_balanced(self, mock_tokenizer):
        """Test class weights for balanced dataset."""
        # Create balanced dataset
        texts = ["text"] * 100
        symptom_indices = [i % NUM_CLASSES for i in range(100)]
        dataset = ReDSM5Dataset(texts, symptom_indices, mock_tokenizer, 512)

        weights = get_class_weights(dataset)

        assert weights.shape == (NUM_CLASSES,)
        assert torch.isfinite(weights).all()
        # For balanced dataset, all weights should be similar
        assert torch.allclose(weights, torch.ones(NUM_CLASSES), rtol=0.1)

    def test_class_weights_imbalanced(self, mock_tokenizer):
        """Test class weights for imbalanced dataset."""
        # Create imbalanced dataset: class 0 appears 90 times, class 1 appears 10 times
        texts = ["text"] * 100
        symptom_indices = [0] * 90 + [1] * 10
        dataset = ReDSM5Dataset(texts, symptom_indices, mock_tokenizer, 512)

        weights = get_class_weights(dataset)

        assert weights.shape == (NUM_CLASSES,)
        # Class 1 (minority) should have higher weight than class 0 (majority)
        assert weights[1] > weights[0]

    def test_class_weights_sum(self, mock_tokenizer):
        """Test that class weights are normalized."""
        texts = ["text"] * 50
        symptom_indices = [0] * 20 + [1] * 30
        dataset = ReDSM5Dataset(texts, symptom_indices, mock_tokenizer, 512)

        weights = get_class_weights(dataset)

        # Weights should be normalized such that sum equals NUM_CLASSES
        assert torch.isclose(weights.sum(), torch.tensor(float(NUM_CLASSES)), rtol=1e-3)

    def test_class_weights_no_zero_division(self, mock_tokenizer):
        """Test that class weights handle missing classes gracefully."""
        # Dataset with only 2 classes present
        texts = ["text"] * 20
        symptom_indices = [0] * 10 + [1] * 10
        dataset = ReDSM5Dataset(texts, symptom_indices, mock_tokenizer, 512)

        weights = get_class_weights(dataset)

        # Should not have inf or nan values
        assert torch.isfinite(weights).all()
        # Weights for missing classes should be low (but not zero due to clamping)
        assert (weights > 0).all()


class TestCreateCVSplits:
    """Tests for create_cv_splits function."""

    @pytest.fixture
    def sample_annotations_file(self):
        """Create a temporary annotations file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Create sample data with all symptom labels
            data = {
                'post_id': list(range(100)),
                'symptom_label': [SYMPTOM_LABELS[i % NUM_CLASSES] for i in range(100)],
            }
            df = pd.DataFrame(data)
            df.to_csv(f.name, index=False)
            return f.name

    def test_create_cv_splits_basic(self, sample_annotations_file):
        """Test basic CV split creation."""
        splits = create_cv_splits(
            annotations_path=sample_annotations_file,
            num_folds=5,
            random_seed=42,
            output_dir=None,
        )

        assert len(splits) == 5
        assert all('train' in split and 'val' in split for split in splits)

    def test_create_cv_splits_sizes(self, sample_annotations_file):
        """Test that split sizes sum to total dataset size."""
        splits = create_cv_splits(
            annotations_path=sample_annotations_file,
            num_folds=5,
            random_seed=42,
            output_dir=None,
        )

        total_samples = 100
        for split in splits:
            train_size = len(split['train'])
            val_size = len(split['val'])
            assert train_size + val_size == total_samples

    def test_create_cv_splits_no_overlap(self, sample_annotations_file):
        """Test that train and val splits don't overlap."""
        splits = create_cv_splits(
            annotations_path=sample_annotations_file,
            num_folds=5,
            random_seed=42,
            output_dir=None,
        )

        for split in splits:
            train_indices = set(split['train'])
            val_indices = set(split['val'])
            # No overlap between train and val
            assert len(train_indices.intersection(val_indices)) == 0

    def test_create_cv_splits_coverage(self, sample_annotations_file):
        """Test that all samples appear in validation exactly once."""
        splits = create_cv_splits(
            annotations_path=sample_annotations_file,
            num_folds=5,
            random_seed=42,
            output_dir=None,
        )

        all_val_indices = set()
        for split in splits:
            all_val_indices.update(split['val'])

        # All samples should appear in validation across folds
        assert len(all_val_indices) == 100
        assert all_val_indices == set(range(100))

    def test_create_cv_splits_reproducibility(self, sample_annotations_file):
        """Test that splits are reproducible with same seed."""
        splits1 = create_cv_splits(
            annotations_path=sample_annotations_file,
            num_folds=5,
            random_seed=42,
            output_dir=None,
        )

        splits2 = create_cv_splits(
            annotations_path=sample_annotations_file,
            num_folds=5,
            random_seed=42,
            output_dir=None,
        )

        # Splits should be identical
        for s1, s2 in zip(splits1, splits2):
            np.testing.assert_array_equal(s1['train'], s2['train'])
            np.testing.assert_array_equal(s1['val'], s2['val'])

    def test_create_cv_splits_different_seeds(self, sample_annotations_file):
        """Test that different seeds produce different splits."""
        splits1 = create_cv_splits(
            annotations_path=sample_annotations_file,
            num_folds=5,
            random_seed=42,
            output_dir=None,
        )

        splits2 = create_cv_splits(
            annotations_path=sample_annotations_file,
            num_folds=5,
            random_seed=123,
            output_dir=None,
        )

        # At least one split should be different
        different = False
        for s1, s2 in zip(splits1, splits2):
            if not np.array_equal(s1['train'], s2['train']):
                different = True
                break

        assert different

    def test_create_cv_splits_with_output_dir(self, sample_annotations_file):
        """Test CV split creation with output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            splits = create_cv_splits(
                annotations_path=sample_annotations_file,
                num_folds=3,
                random_seed=42,
                output_dir=tmpdir,
            )

            # Check that files are created
            output_path = Path(tmpdir)
            assert (output_path / 'split_metadata.json').exists()

            for fold_idx in range(3):
                assert (output_path / f'fold_{fold_idx}_train.csv').exists()
                assert (output_path / f'fold_{fold_idx}_val.csv').exists()
                assert (output_path / f'fold_{fold_idx}_metadata.json').exists()


class TestGetFoldStatistics:
    """Tests for get_fold_statistics function."""

    def test_get_fold_statistics_basic(self):
        """Test basic fold statistics computation."""
        splits = [
            {'train': np.array([0, 1, 2, 3, 4, 5, 6, 7]), 'val': np.array([8, 9])},
            {'train': np.array([0, 1, 2, 3, 4, 5, 8, 9]), 'val': np.array([6, 7])},
        ]

        stats = get_fold_statistics(splits)

        assert len(stats) == 2
        assert 'fold' in stats.columns
        assert 'train_size' in stats.columns
        assert 'val_size' in stats.columns
        assert 'total_size' in stats.columns

    def test_get_fold_statistics_sizes(self):
        """Test fold statistics compute correct sizes."""
        splits = [
            {'train': np.array([0, 1, 2, 3, 4, 5, 6, 7]), 'val': np.array([8, 9])},
        ]

        stats = get_fold_statistics(splits)

        assert stats.iloc[0]['train_size'] == 8
        assert stats.iloc[0]['val_size'] == 2
        assert stats.iloc[0]['total_size'] == 10

    def test_get_fold_statistics_ratios(self):
        """Test fold statistics compute correct ratios."""
        splits = [
            {'train': np.array([0, 1, 2, 3, 4, 5, 6, 7]), 'val': np.array([8, 9])},
        ]

        stats = get_fold_statistics(splits)

        assert abs(stats.iloc[0]['train_ratio'] - 0.8) < 0.01
        assert abs(stats.iloc[0]['val_ratio'] - 0.2) < 0.01


class TestDataIntegration:
    """Integration tests for data pipeline."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.return_value = {
            'input_ids': torch.randint(0, 1000, (1, 512)),
            'attention_mask': torch.ones(1, 512),
        }
        return tokenizer

    def test_dataset_to_dataloader(self, mock_tokenizer):
        """Test that dataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        texts = ["text"] * 10
        symptom_indices = [0] * 10
        dataset = ReDSM5Dataset(texts, symptom_indices, mock_tokenizer, 512)

        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        batch = next(iter(dataloader))
        assert batch['input_ids'].shape == (2, 512)
        assert batch['attention_mask'].shape == (2, 512)
        assert batch['symptom_idx'].shape == (2,)

    def test_class_weights_with_dataloader(self, mock_tokenizer):
        """Test that class weights can be used with loss function."""
        import torch.nn as nn

        texts = ["text"] * 20
        symptom_indices = [0] * 15 + [1] * 5
        dataset = ReDSM5Dataset(texts, symptom_indices, mock_tokenizer, 512)

        weights = get_class_weights(dataset)
        criterion = nn.CrossEntropyLoss(weight=weights)

        # Test that criterion works
        logits = torch.randn(4, NUM_CLASSES)
        labels = torch.tensor([0, 1, 0, 1])
        loss = criterion(logits, labels)

        assert torch.isfinite(loss)
        assert loss.requires_grad


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
