#!/usr/bin/env python
"""
Benchmark training performance across different configurations.
"""

import subprocess
import time
import json
from pathlib import Path


def run_benchmark(config_name, num_epochs=2):
    """Run training with a specific config and measure performance."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {config_name}")
    print(f"{'='*60}\n")

    cmd = [
        "python", "src/training/train_nli_binary.py",
        f"experiment={config_name}",
        f"training.num_epochs={num_epochs}",
        "data.post_limit=500",  # Limit dataset for faster benchmarking
        "mlflow.enabled=false"
    ]

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_time = time.time() - start_time

    return {
        'config': config_name,
        'time': elapsed_time,
        'success': result.returncode == 0,
        'output': result.stdout
    }


def main():
    """Run benchmarks for all configurations."""
    configs = [
        'rtx3090_optimized',  # Base optimized
        'rtx3090_max_perf',   # Max performance
        'rtx3090_ultra'       # Ultra with all optional optimizations
    ]

    results = []
    for config in configs:
        result = run_benchmark(config)
        results.append(result)

        print(f"\nConfig: {config}")
        print(f"Time: {result['time']:.2f}s")
        print(f"Success: {result['success']}")

    # Save results
    output_dir = Path("outputs/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "benchmark_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}\n")

    for result in results:
        print(f"{result['config']:20s}: {result['time']:6.2f}s")

    # Calculate speedups
    if len(results) > 1:
        baseline = results[0]['time']
        print(f"\nSpeedups vs {results[0]['config']}:")
        for result in results[1:]:
            speedup = baseline / result['time']
            print(f"  {result['config']:20s}: {speedup:.2f}x")


if __name__ == "__main__":
    main()
