"""
Length Bucketing for Efficient DataLoader.

Groups samples by similar sequence lengths to minimize padding overhead.
"""

import torch
from torch.utils.data import Sampler
from typing import Iterator, List
import numpy as np


class LengthBucketSampler(Sampler):
    """
    Groups samples into buckets by sequence length, then samples from buckets.

    This reduces padding waste by batching similar-length sequences together.

    Args:
        lengths: List of sequence lengths for each sample
        batch_size: Target batch size
        drop_last: Whether to drop incomplete batches
        shuffle: Whether to shuffle within buckets
    """

    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        num_buckets: int = 10
    ):
        self.lengths = np.array(lengths)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.num_buckets = num_buckets

        # Create buckets based on length quantiles
        self.bucket_boundaries = np.percentile(
            self.lengths,
            np.linspace(0, 100, num_buckets + 1)
        )

        # Assign each sample to a bucket
        self.bucket_indices = np.digitize(self.lengths, self.bucket_boundaries[1:-1])

        # Group indices by bucket
        self.buckets = {}
        for idx in range(len(self.lengths)):
            bucket_id = self.bucket_indices[idx]
            if bucket_id not in self.buckets:
                self.buckets[bucket_id] = []
            self.buckets[bucket_id].append(idx)

    def __iter__(self) -> Iterator[List[int]]:
        """Iterate over batches."""
        # Shuffle within each bucket if requested
        if self.shuffle:
            for bucket_id in self.buckets:
                np.random.shuffle(self.buckets[bucket_id])

        # Collect all batches
        all_batches = []
        for bucket_id in self.buckets:
            bucket = self.buckets[bucket_id]

            # Create batches from this bucket
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]

                # Skip incomplete batches if drop_last
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)

        # Shuffle batches if requested
        if self.shuffle:
            np.random.shuffle(all_batches)

        # Yield batches
        for batch in all_batches:
            yield batch

    def __len__(self) -> int:
        """Number of batches."""
        total_batches = 0
        for bucket_id in self.buckets:
            bucket_size = len(self.buckets[bucket_id])
            if self.drop_last:
                total_batches += bucket_size // self.batch_size
            else:
                total_batches += (bucket_size + self.batch_size - 1) // self.batch_size
        return total_batches


def collate_fn_dynamic_padding(batch):
    """
    Collate function with dynamic padding to max length in batch.

    Instead of padding all sequences to max_length (e.g., 512),
    pad only to the longest sequence in this batch.
    """
    # Find max length in this batch
    max_len = max(len(item['input_ids']) for item in batch)

    # Pad to max_len (not global max_length)
    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        seq_len = len(item['input_ids'])
        padding_len = max_len - seq_len

        # Pad input_ids
        padded_input = item['input_ids'] + [0] * padding_len
        input_ids.append(padded_input)

        # Pad attention_mask
        padded_mask = item['attention_mask'] + [0] * padding_len
        attention_mask.append(padded_mask)

        # Labels don't need padding
        labels.append(item['labels'])

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long)
    }
