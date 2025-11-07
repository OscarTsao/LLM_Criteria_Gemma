## Benchmarks

Performance benchmarks for LLM_Criteria_Gemma components.

### Available Benchmarks

#### 1. Pooler Benchmarks

Measure performance of different pooling strategies:

```bash
# CPU benchmarks
python benchmarks/benchmark_poolers.py --device cpu

# GPU benchmarks
python benchmarks/benchmark_poolers.py --device cuda

# Custom configuration
python benchmarks/benchmark_poolers.py \
    --device cuda \
    --batch-sizes 1 4 8 16 32 64 \
    --seq-lengths 128 256 512 1024 \
    --hidden-dim 2048 \
    --iterations 100
```

**Metrics:**
- Average inference time (ms)
- Throughput (samples/sec)
- Performance vs batch size
- Performance vs sequence length

#### 2. Data Loading Benchmarks

Measure data loading and tokenization performance:

```bash
python benchmarks/benchmark_data_loading.py \
    --num-samples 1000 \
    --batch-sizes 8 16 32 64 \
    --num-workers 4
```

**Metrics:**
- Tokenization speed
- DataLoader throughput
- Memory usage
- Batch processing time

### Results

Benchmark results are saved to `benchmarks/results/`:

```
benchmarks/results/
├── pooler_benchmarks.csv      # Detailed pooler results
├── pooler_benchmarks.json     # JSON summary
└── data_loading_benchmarks.json
```

### Interpreting Results

#### Pooler Performance

- **MeanPooler**: Typically fastest for small batch sizes
- **CLSPooler**: Fastest overall (single token extraction)
- **AttentionPooler**: Slowest (additional compute for attention)

**Expected Performance** (CPU, batch_size=16, seq_length=512):
- CLSPooler: ~0.1 ms
- MeanPooler: ~0.5 ms
- MaxPooler: ~0.6 ms
- AttentionPooler: ~2.0 ms

#### Optimization Tips

1. **Batch Size**: Larger batches improve throughput but increase latency
2. **Sequence Length**: Longer sequences slow down all poolers
3. **Device**: GPU provides 10-100x speedup for large batches
4. **Pooler Choice**: Use CLSPooler if appropriate for your task

### Adding New Benchmarks

Create a new benchmark file in `benchmarks/`:

```python
"""Benchmark description."""

def benchmark_component(config):
    # Warmup
    for _ in range(warmup_iterations):
        run_component(config)

    # Measure
    start_time = time.perf_counter()
    for _ in range(iterations):
        run_component(config)
    end_time = time.perf_counter()

    return calculate_metrics(start_time, end_time)

if __name__ == '__main__':
    main()
```

### Continuous Benchmarking

Benchmarks are run automatically in CI/CD on:
- Pull requests (quick benchmarks)
- Nightly builds (comprehensive benchmarks)
- Release tags (full benchmark suite)

### Comparing Results

```python
import pandas as pd

# Load benchmark results
current = pd.read_csv('benchmarks/results/pooler_benchmarks.csv')
baseline = pd.read_csv('benchmarks/baseline/pooler_benchmarks.csv')

# Compare performance
comparison = current.merge(baseline, on=['pooler', 'batch_size'], suffixes=('_current', '_baseline'))
comparison['speedup'] = comparison['avg_time_ms_baseline'] / comparison['avg_time_ms_current']

print(comparison[['pooler', 'batch_size', 'speedup']])
```
