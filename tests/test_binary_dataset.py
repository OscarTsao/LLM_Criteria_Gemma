"""
Tests for binary classification dataset.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data.binary_dataset import (
    BinaryReDSM5Dataset,
    CRITERION_DESCRIPTIONS,
    CRITERION_LABELS,
    get_binary_class_weights
)


class TestBinaryDataset:
    """Test binary classification dataset."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Mock tokenizer for testing."""
        class MockTokenizer:
            def __call__(self, text1, text2, max_length, padding, truncation, return_tensors):
                import torch
                # Return mock encoding
                seq_len = min(max_length, 50)
                return {
                    'input_ids': torch.randint(0, 1000, (1, seq_len)),
                    'attention_mask': torch.ones(1, seq_len),
                }

            pad_token = '[PAD]'
            eos_token = '[EOS]'

        return MockTokenizer()

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        posts = [
            "I can't sleep at night",
            "Nothing makes me happy anymore",
            "I feel so worthless",
        ]
        symptom_indices = [3, 1, 6]  # SLEEP_ISSUES, ANHEDONIA, WORTHLESSNESS

        return posts, symptom_indices

    def test_binary_dataset_creation_all_negatives(self, mock_tokenizer, sample_data):
        """Test dataset creation with 'all' negative sampling."""
        posts, symptom_indices = sample_data

        dataset = BinaryReDSM5Dataset(
            posts, symptom_indices, mock_tokenizer,
            max_length=512,
            negative_sampling='all'
        )

        # Should create 3 posts × 10 criteria = 30 pairs
        assert len(dataset) == 30

        # Check class distribution
        dist = dataset.get_class_distribution()
        assert dist['positive'] == 3  # 1 positive per post
        assert dist['negative'] == 27  # 9 negatives per post
        assert dist['total'] == 30

    def test_binary_dataset_creation_random_negatives(self, mock_tokenizer, sample_data):
        """Test dataset creation with 'random' negative sampling."""
        posts, symptom_indices = sample_data

        dataset = BinaryReDSM5Dataset(
            posts, symptom_indices, mock_tokenizer,
            max_length=512,
            negative_sampling='random',
            num_negatives=2
        )

        # Should create 3 posts × (1 pos + 2 neg) = 9 pairs
        assert len(dataset) == 9

        # Check class distribution
        dist = dataset.get_class_distribution()
        assert dist['positive'] == 3  # 1 positive per post
        assert dist['negative'] == 6  # 2 negatives per post

    def test_binary_dataset_pairs_are_correct(self, mock_tokenizer, sample_data):
        """Test that positive pairs match ground truth."""
        posts, symptom_indices = sample_data

        dataset = BinaryReDSM5Dataset(
            posts, symptom_indices, mock_tokenizer,
            max_length=512,
            negative_sampling='all'
        )

        # Check that each post has exactly 1 positive pair with correct criterion
        for post_idx, true_symptom_idx in enumerate(symptom_indices):
            # Find all pairs for this post
            post_pairs = [s for s in dataset.samples if s[3] == post_idx]

            # Should have 10 pairs (1 pos, 9 neg)
            assert len(post_pairs) == 10

            # Find positive pair
            positive_pairs = [s for s in post_pairs if s[2] == 1]
            assert len(positive_pairs) == 1

            # Check that positive pair has correct criterion
            post_text, criterion_text, label, _ = positive_pairs[0]
            true_criterion_name = CRITERION_LABELS[true_symptom_idx]
            true_criterion_desc = CRITERION_DESCRIPTIONS[true_criterion_name]

            assert criterion_text == true_criterion_desc
            assert label == 1

    def test_binary_dataset_getitem(self, mock_tokenizer, sample_data):
        """Test __getitem__ returns correct format."""
        posts, symptom_indices = sample_data

        dataset = BinaryReDSM5Dataset(
            posts, symptom_indices, mock_tokenizer,
            max_length=512,
            negative_sampling='all'
        )

        item = dataset[0]

        # Check keys
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'label' in item
        assert 'post_idx' in item

        # Check shapes
        assert item['input_ids'].dim() == 1
        assert item['attention_mask'].dim() == 1
        assert item['label'].dim() == 0  # Scalar
        assert item['post_idx'].dim() == 0  # Scalar

    def test_class_imbalance_with_all_sampling(self, mock_tokenizer, sample_data):
        """Test that 'all' sampling creates 1:9 imbalance."""
        posts, symptom_indices = sample_data

        dataset = BinaryReDSM5Dataset(
            posts, symptom_indices, mock_tokenizer,
            max_length=512,
            negative_sampling='all'
        )

        dist = dataset.get_class_distribution()

        # Ratio should be 1:9
        expected_balance = 3 / 30  # 0.1
        assert abs(dist['balance'] - expected_balance) < 0.01

    def test_class_weights_computation(self, mock_tokenizer, sample_data):
        """Test class weight computation for imbalanced dataset."""
        posts, symptom_indices = sample_data

        dataset = BinaryReDSM5Dataset(
            posts, symptom_indices, mock_tokenizer,
            max_length=512,
            negative_sampling='all'
        )

        weights = get_binary_class_weights(dataset)

        # Should have 2 weights
        assert len(weights) == 2

        # Positive class should have higher weight (since it's minority)
        assert weights[1] > weights[0]

        # Weights should be positive
        assert weights[0] > 0
        assert weights[1] > 0

    def test_criterion_descriptions_exist(self):
        """Test that all criteria have descriptions."""
        assert len(CRITERION_DESCRIPTIONS) == 10
        assert len(CRITERION_LABELS) == 10

        for label in CRITERION_LABELS:
            assert label in CRITERION_DESCRIPTIONS
            assert len(CRITERION_DESCRIPTIONS[label]) > 0

    def test_no_duplicate_pairs(self, mock_tokenizer, sample_data):
        """Test that no duplicate pairs are created."""
        posts, symptom_indices = sample_data

        dataset = BinaryReDSM5Dataset(
            posts, symptom_indices, mock_tokenizer,
            max_length=512,
            negative_sampling='all'
        )

        # Extract (post, criterion, label) tuples
        pair_tuples = [(s[0], s[1], s[2]) for s in dataset.samples]

        # Check no duplicates
        assert len(pair_tuples) == len(set(pair_tuples))

    def test_balanced_sampling_with_random(self, mock_tokenizer):
        """Test that 'random' sampling can create balanced dataset."""
        posts = ["Sample post"] * 10
        symptom_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # All 10 symptoms

        dataset = BinaryReDSM5Dataset(
            posts, symptom_indices, mock_tokenizer,
            max_length=512,
            negative_sampling='random',
            num_negatives=9  # 1:9 ratio like 'all'
        )

        dist = dataset.get_class_distribution()

        # Should be similar to 'all' sampling
        assert dist['positive'] == 10
        assert dist['negative'] == 90

    def test_single_negative_sampling(self, mock_tokenizer, sample_data):
        """Test extreme case: only 1 negative per positive."""
        posts, symptom_indices = sample_data

        dataset = BinaryReDSM5Dataset(
            posts, symptom_indices, mock_tokenizer,
            max_length=512,
            negative_sampling='random',
            num_negatives=1
        )

        # 3 posts × 2 pairs = 6 total
        assert len(dataset) == 6

        dist = dataset.get_class_distribution()
        assert dist['positive'] == 3
        assert dist['negative'] == 3
        assert dist['balance'] == 0.5  # Perfectly balanced


class TestCriterionDescriptions:
    """Test criterion descriptions."""

    def test_all_symptoms_have_descriptions(self):
        """Test that all 10 symptoms have descriptions."""
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

        for symptom in expected_symptoms:
            assert symptom in CRITERION_DESCRIPTIONS

    def test_descriptions_are_informative(self):
        """Test that descriptions are not empty and informative."""
        for label, description in CRITERION_DESCRIPTIONS.items():
            # Description should be a non-empty string
            assert isinstance(description, str)
            assert len(description) > 10  # At least some content

            # Should contain the symptom keyword (rough heuristic)
            label_words = label.lower().split('_')
            desc_lower = description.lower()

            # At least one word from label should appear in description
            # (relaxed check for informativeness)
            assert any(word in desc_lower for word in label_words if len(word) > 3)
