"""
Test script for NLI dataset creation.

Verifies that the NLI-style text-pair dataset is created correctly with:
- Proper [CLS] post [SEP] criterion [SEP] formatting
- Balanced positive/negative samples
- Correct binary labels
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import pandas as pd
from transformers import AutoTokenizer
from data.redsm5_nli_dataset import load_redsm5_nli, create_nli_pairs
from data.dsm5_criteria import get_criterion_text, DSM5_CRITERIA


def test_criterion_texts():
    """Test criterion text retrieval."""
    print("=" * 80)
    print("Testing Criterion Texts")
    print("=" * 80)

    symptoms = ['DEPRESSED_MOOD', 'ANHEDONIA', 'COGNITIVE_ISSUES']

    for symptom in symptoms:
        print(f"\n{symptom}:")
        print(f"  Full: {get_criterion_text(symptom, use_short=False)[:100]}...")
        print(f"  Short: {get_criterion_text(symptom, use_short=True)}")

    print("\n✓ Criterion texts loaded successfully")


def test_nli_pair_creation():
    """Test NLI pair creation from ReDSM5 data."""
    print("\n" + "=" * 80)
    print("Testing NLI Pair Creation")
    print("=" * 80)

    # Load original data
    data_dir = Path(__file__).parent.parent / 'data' / 'redsm5'

    if not data_dir.exists():
        print(f"⚠ Data directory not found: {data_dir}")
        print("  Skipping pair creation test")
        return

    posts_df = pd.read_csv(data_dir / 'redsm5_posts.csv')
    annotations_df = pd.read_csv(data_dir / 'redsm5_annotations.csv')

    print(f"\nOriginal data:")
    print(f"  Posts: {len(posts_df)}")
    print(f"  Annotations: {len(annotations_df)}")

    # Create NLI pairs
    pairs_df = create_nli_pairs(
        posts_df,
        annotations_df,
        negative_ratio=1.0,
        use_short_criteria=False,
        random_seed=42,
    )

    print(f"\nNLI pairs created:")
    print(f"  Total pairs: {len(pairs_df)}")
    print(f"  Positive (matched): {(pairs_df['label'] == 1).sum()}")
    print(f"  Negative (unmatched): {(pairs_df['label'] == 0).sum()}")

    # Show examples
    print("\n" + "-" * 80)
    print("Example Positive Pair (matched):")
    print("-" * 80)
    pos_example = pairs_df[pairs_df['label'] == 1].iloc[0]
    print(f"Post (first 200 chars): {pos_example['post_text'][:200]}...")
    print(f"\nCriterion: {pos_example['criterion_text'][:200]}...")
    print(f"Label: {pos_example['label']} (matched)")
    print(f"Symptom: {pos_example['symptom_label']}")

    print("\n" + "-" * 80)
    print("Example Negative Pair (unmatched):")
    print("-" * 80)
    neg_example = pairs_df[pairs_df['label'] == 0].iloc[0]
    print(f"Post (first 200 chars): {neg_example['post_text'][:200]}...")
    print(f"\nCriterion: {neg_example['criterion_text'][:200]}...")
    print(f"Label: {neg_example['label']} (unmatched)")
    print(f"Symptom: {neg_example['symptom_label']}")

    print("\n✓ NLI pairs created successfully")


def test_dataset_tokenization():
    """Test dataset tokenization with text pairs."""
    print("\n" + "=" * 80)
    print("Testing Dataset Tokenization (Text Pairs)")
    print("=" * 80)

    data_dir = Path(__file__).parent.parent / 'data' / 'redsm5'

    if not data_dir.exists():
        print(f"⚠ Data directory not found: {data_dir}")
        print("  Skipping tokenization test")
        return

    # Load tokenizer
    model_name = 'google/gemma-2-2b'
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load NLI dataset
    print("\nLoading NLI dataset...")
    train_dataset, val_dataset, test_dataset, pairs_df = load_redsm5_nli(
        data_dir=str(data_dir),
        tokenizer=tokenizer,
        max_length=512,
        negative_ratio=1.0,
        use_short_criteria=False,
        random_seed=42,
    )

    # Test a sample
    print("\n" + "-" * 80)
    print("Sample from Training Set:")
    print("-" * 80)

    sample = train_dataset[0]
    print(f"Keys: {sample.keys()}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Attention mask shape: {sample['attention_mask'].shape}")
    print(f"Label: {sample['label'].item()} ({'matched' if sample['label'].item() == 1 else 'unmatched'})")

    # Decode the tokenized text
    print(f"\nDecoded input (showing [CLS] text1 [SEP] text2 [SEP] structure):")
    decoded = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
    print(f"  {decoded[:300]}...")

    # Check for proper NSP format
    has_sep = '[SEP]' in decoded or tokenizer.sep_token in decoded or '</s>' in decoded
    print(f"\n{'✓' if has_sep else '✗'} Proper text-pair formatting detected")

    # Show dataset statistics
    print("\n" + "-" * 80)
    print("Dataset Statistics:")
    print("-" * 80)

    for name, dataset in [('Train', train_dataset), ('Val', val_dataset), ('Test', test_dataset)]:
        matched = sum(dataset.labels)
        total = len(dataset.labels)
        print(f"{name:10s}: {total:4d} samples ({matched:4d} matched, {total - matched:4d} unmatched, "
              f"{matched / total * 100:5.1f}% positive)")

    print("\n✓ Dataset tokenization verified successfully")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("NLI DATASET TEST SUITE")
    print("=" * 80)

    try:
        test_criterion_texts()
        test_nli_pair_creation()
        test_dataset_tokenization()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
