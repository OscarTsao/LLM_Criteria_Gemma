"""
Benchmark data loading pipeline performance.

Measures data loading, tokenization, and DataLoader throughput.
"""

import torch
from torch.utils.data import DataLoader
import time
import sys
from pathlib import Path
import pandas as pd
import json
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def benchmark_tokenization(texts: List[str], tokenizer, max_length: int = 512, iterations: int = 10):
    """
    Benchmark tokenization speed.

    Args:
        texts: List of text strings
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        iterations: Number of iterations

    Returns:
        Dict with benchmark results
    """
    # Warmup
    for _ in range(3):
        _ = tokenizer(texts[:10], max_length=max_length, padding='max_length', truncation=True)

    # Benchmark
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = tokenizer(texts, max_length=max_length, padding='max_length', truncation=True)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time = total_time / iterations
    throughput = (len(texts) * iterations) / total_time

    return {
        'num_texts': len(texts),
        'max_length': max_length,
        'avg_time_ms': avg_time * 1000,
        'throughput_texts_per_sec': throughput,
    }


def benchmark_dataloader(dataset, batch_sizes: List[int], num_workers: int = 0):
    """
    Benchmark DataLoader throughput.

    Args:
        dataset: PyTorch Dataset
        batch_sizes: List of batch sizes to test
        num_workers: Number of worker processes

    Returns:
        List of benchmark results
    """
    results = []

    for batch_size in batch_sizes:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        # Warmup
        for i, batch in enumerate(dataloader):
            if i >= 5:
                break

        # Benchmark
        start_time = time.perf_counter()
        num_batches = 0
        num_samples = 0

        for batch in dataloader:
            num_batches += 1
            num_samples += len(batch['input_ids'])

        end_time = time.perf_counter()

        total_time = end_time - start_time
        throughput = num_samples / total_time

        results.append({
            'batch_size': batch_size,
            'num_workers': num_workers,
            'num_batches': num_batches,
            'num_samples': num_samples,
            'total_time_sec': total_time,
            'throughput_samples_per_sec': throughput,
            'time_per_batch_ms': (total_time / num_batches) * 1000,
        })

    return results


def benchmark_memory_usage():
    """
    Benchmark memory usage of data structures.

    Returns:
        Dict with memory usage statistics
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Measure CUDA memory
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)

        return {
            'cuda_allocated_gb': allocated,
            'cuda_reserved_gb': reserved,
        }
    else:
        # CPU memory measurement (approximate)
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            'cpu_rss_gb': memory_info.rss / (1024**3),
            'cpu_vms_gb': memory_info.vms / (1024**3),
        }


def main():
    """Main benchmark runner."""
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark data loading')
    parser.add_argument('--num-samples', type=int, default=1000)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[8, 16, 32, 64])
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--output', type=str, default='benchmarks/results/data_loading_benchmarks.json')
    args = parser.parse_args()

    print("=" * 80)
    print("DATA LOADING BENCHMARK")
    print("=" * 80)
    print(f"Num samples: {args.num_samples}")
    print(f"Max length: {args.max_length}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Num workers: {args.num_workers}")

    # Create synthetic data
    texts = [f"This is sample text number {i} for benchmarking purposes." for i in range(args.num_samples)]

    # Note: Actual benchmarking requires tokenizer which downloads models
    # This is a structure for when models are available

    results = {
        'config': {
            'num_samples': args.num_samples,
            'max_length': args.max_length,
            'batch_sizes': args.batch_sizes,
            'num_workers': args.num_workers,
        },
        'memory': benchmark_memory_usage(),
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
