"""
Benchmark pooling strategies for performance comparison.

Measures inference time, memory usage, and throughput for different poolers.
"""

import torch
import time
import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.poolers import (
    MeanPooler,
    CLSPooler,
    MaxPooler,
    FirstKPooler,
    LastKPooler,
    AttentionPooler,
)


def benchmark_pooler(
    pooler,
    batch_sizes: List[int],
    seq_lengths: List[int],
    hidden_dim: int = 768,
    num_iterations: int = 100,
    device: str = 'cpu',
):
    """
    Benchmark a pooler across different batch sizes and sequence lengths.

    Args:
        pooler: Pooler instance to benchmark
        batch_sizes: List of batch sizes to test
        seq_lengths: List of sequence lengths to test
        hidden_dim: Hidden dimension size
        num_iterations: Number of iterations for averaging
        device: Device to run on ('cpu' or 'cuda')

    Returns:
        Dictionary of benchmark results
    """
    results = []

    for batch_size in batch_sizes:
        for seq_length in seq_lengths:
            # Create random inputs
            hidden_states = torch.randn(batch_size, seq_length, hidden_dim, device=device)
            attention_mask = torch.ones(batch_size, seq_length, device=device)

            # Move pooler to device if it's a module
            if isinstance(pooler, torch.nn.Module):
                pooler = pooler.to(device)

            # Warmup
            for _ in range(10):
                _ = pooler(hidden_states, attention_mask)

            # Benchmark
            if device == 'cuda':
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            for _ in range(num_iterations):
                _ = pooler(hidden_states, attention_mask)

            if device == 'cuda':
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            # Calculate metrics
            total_time = end_time - start_time
            avg_time = total_time / num_iterations
            throughput = (batch_size * num_iterations) / total_time

            results.append({
                'pooler': pooler.__class__.__name__,
                'batch_size': batch_size,
                'seq_length': seq_length,
                'hidden_dim': hidden_dim,
                'avg_time_ms': avg_time * 1000,
                'throughput_samples_per_sec': throughput,
                'device': device,
            })

    return results


def run_all_benchmarks(
    batch_sizes: List[int] = [1, 4, 8, 16, 32],
    seq_lengths: List[int] = [128, 256, 512],
    hidden_dim: int = 768,
    num_iterations: int = 100,
    device: str = 'cpu',
):
    """
    Run benchmarks for all pooler types.

    Args:
        batch_sizes: Batch sizes to test
        seq_lengths: Sequence lengths to test
        hidden_dim: Hidden dimension
        num_iterations: Number of iterations
        device: Device to run on

    Returns:
        DataFrame with all results
    """
    all_results = []

    poolers = [
        ('MeanPooler', MeanPooler()),
        ('CLSPooler', CLSPooler()),
        ('MaxPooler', MaxPooler()),
        ('FirstKPooler(k=1)', FirstKPooler(k=1)),
        ('LastKPooler(k=1)', LastKPooler(k=1)),
        ('AttentionPooler', AttentionPooler(hidden_dim)),
    ]

    for name, pooler in poolers:
        print(f"Benchmarking {name}...")
        results = benchmark_pooler(
            pooler, batch_sizes, seq_lengths, hidden_dim, num_iterations, device
        )
        all_results.extend(results)

    return pd.DataFrame(all_results)


def print_benchmark_summary(df: pd.DataFrame):
    """Print a formatted summary of benchmark results."""
    print("\n" + "=" * 80)
    print("POOLER BENCHMARK SUMMARY")
    print("=" * 80)

    # Group by pooler and show average performance
    summary = df.groupby('pooler').agg({
        'avg_time_ms': ['mean', 'std'],
        'throughput_samples_per_sec': ['mean', 'std'],
    }).round(4)

    print("\nAverage Performance by Pooler:")
    print(summary)

    # Find fastest and slowest
    avg_by_pooler = df.groupby('pooler')['avg_time_ms'].mean().sort_values()

    print(f"\nFastest Pooler: {avg_by_pooler.index[0]} ({avg_by_pooler.iloc[0]:.4f} ms)")
    print(f"Slowest Pooler: {avg_by_pooler.index[-1]} ({avg_by_pooler.iloc[-1]:.4f} ms)")

    # Performance vs batch size
    print("\nPerformance vs Batch Size:")
    for pooler in df['pooler'].unique():
        pooler_df = df[df['pooler'] == pooler]
        batch_perf = pooler_df.groupby('batch_size')['avg_time_ms'].mean()
        print(f"  {pooler}: {batch_perf.to_dict()}")

    print("=" * 80)


def main():
    """Main benchmark runner."""
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark pooling strategies')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4, 8, 16, 32])
    parser.add_argument('--seq-lengths', type=int, nargs='+', default=[128, 256, 512])
    parser.add_argument('--hidden-dim', type=int, default=768)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--output', type=str, default='benchmarks/results/pooler_benchmarks.csv')
    args = parser.parse_args()

    print(f"Running benchmarks on {args.device}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Sequence lengths: {args.seq_lengths}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Iterations: {args.iterations}")

    # Run benchmarks
    results_df = run_all_benchmarks(
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        hidden_dim=args.hidden_dim,
        num_iterations=args.iterations,
        device=args.device,
    )

    # Print summary
    print_benchmark_summary(results_df)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Save JSON summary
    json_path = output_path.with_suffix('.json')
    summary_dict = {
        'device': args.device,
        'config': {
            'batch_sizes': args.batch_sizes,
            'seq_lengths': args.seq_lengths,
            'hidden_dim': args.hidden_dim,
            'iterations': args.iterations,
        },
        'results': results_df.to_dict(orient='records'),
    }

    with open(json_path, 'w') as f:
        json.dump(summary_dict, f, indent=2)

    print(f"JSON summary saved to: {json_path}")


if __name__ == '__main__':
    main()
