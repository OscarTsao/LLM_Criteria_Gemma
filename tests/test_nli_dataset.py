"""
Test script for Binary NLI Dataset.

Verifies that the NLI dataset loading and tokenization works correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from transformers import AutoTokenizer
from data.redsm5_nli_dataset import (
    load_redsm5_nli,
    get_label_distribution,
    get_criterion_distribution
)
from data.dsm5_criteria import get_all_criteria, get_symptom_labels


def test_criteria_loading():
    """Test that DSM-5 criteria are loaded correctly."""
    print("Testing DSM-5 criteria loading...")

    criteria = get_all_criteria()
    symptom_labels = get_symptom_labels()

    assert len(criteria) == 9, f"Expected 9 criteria, got {len(criteria)}"
    assert len(symptom_labels) == 9, f"Expected 9 symptom labels, got {len(symptom_labels)}"

    print(f"✓ Loaded {len(criteria)} criteria")
    for label in symptom_labels:
        print(f"  - {label}: {criteria[label][:50]}...")

    print()


def test_dataset_loading():
    """Test that NLI dataset loads correctly."""
    print("Testing NLI dataset loading...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")  # Use smaller model for testing

    # Load dataset
    data_dir = "data/redsm5"
    train_dataset, val_dataset, test_dataset = load_redsm5_nli(
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=512,
        test_size=0.15,
        val_size=0.15,
        random_seed=42,
    )

    print(f"\n✓ Dataset loaded successfully:")
    print(f"  Train: {len(train_dataset)} pairs")
    print(f"  Val: {len(val_dataset)} pairs")
    print(f"  Test: {len(test_dataset)} pairs")

    return train_dataset, val_dataset, test_dataset


def test_dataset_samples(dataset):
    """Test individual samples from the dataset."""
    print("\nTesting dataset samples...")

    # Get a few samples
    sample = dataset[0]

    print(f"✓ Sample structure:")
    print(f"  Keys: {sample.keys()}")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  attention_mask shape: {sample['attention_mask'].shape}")
    print(f"  label: {sample['labels'].item()}")
    print(f"  post_id: {sample['post_id']}")
    print(f"  criterion_name: {sample['criterion_name']}")

    # Check for both matched and unmatched examples
    labels = [dataset[i]['labels'].item() for i in range(min(100, len(dataset)))]
    num_matched = sum(labels)
    print(f"\n✓ Label distribution in first 100 samples:")
    print(f"  Matched (1): {num_matched}")
    print(f"  Unmatched (0): {len(labels) - num_matched}")


def test_label_distribution(dataset):
    """Test label distribution analysis."""
    print("\nTesting label distribution...")

    dist = get_label_distribution(dataset)
    print(f"✓ Label distribution:")
    print(dist)


def test_criterion_distribution(dataset):
    """Test per-criterion distribution analysis."""
    print("\nTesting criterion distribution...")

    dist = get_criterion_distribution(dataset)
    print(f"✓ Criterion distribution:")
    print(dist)


def main():
    """Run all tests."""
    print("=" * 60)
    print("Binary NLI Dataset Test Suite")
    print("=" * 60)
    print()

    try:
        # Test 1: Criteria loading
        test_criteria_loading()

        # Test 2: Dataset loading
        train_dataset, val_dataset, test_dataset = test_dataset_loading()

        # Test 3: Sample structure
        test_dataset_samples(train_dataset)

        # Test 4: Label distribution
        test_label_distribution(train_dataset)

        # Test 5: Criterion distribution
        test_criterion_distribution(train_dataset)

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\n✗ Error: Data files not found")
        print(f"  {e}")
        print("\nMake sure the ReDSM5 dataset is in data/redsm5/")
        return 1

    except Exception as e:
        print(f"\n✗ Error during testing:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
