# Hardware Optimization Guide

This guide explains how to optimize training performance for different GPU configurations based on PyTorch best practices and the Gemma Encoder paper recommendations.

## Table of Contents

- [Quick Start](#quick-start)
- [Automatic Hardware Detection](#automatic-hardware-detection)
- [Hardware-Specific Configurations](#hardware-specific-configurations)
- [Optimization Techniques](#optimization-techniques)
- [Makefile Commands](#makefile-commands)
- [Manual Configuration](#manual-configuration)
- [Benchmarking](#benchmarking)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Check Your Hardware

```bash
# Show detailed hardware information with recommendations
make check-hardware
```

This will display:
- GPU name and memory
- Compute capability
- BFloat16 and TF32 support
- Recommended batch size and settings

### 2. Train with Optimal Settings

```bash
# Automatic hardware detection (recommended)
make nli-train-auto

# Or use a specific GPU profile:
make nli-train-4090      # RTX 4090 (24GB)
make nli-train-3090      # RTX 3090 (24GB)
make nli-train-low-mem   # Low-memory GPUs (8-12GB)
```

---

## Automatic Hardware Detection

The hardware optimizer automatically detects your GPU and applies optimal settings:

```python
from src.utils.hardware_optimizer import (
    detect_gpu_info,
    optimize_pytorch_settings,
    print_hardware_info
)

# Print hardware information
print_hardware_info()

# Get recommended configuration
info = detect_gpu_info()
print(f"GPU: {info['gpu_name']}")
print(f"Memory: {info['gpu_memory_gb']:.2f} GB")
print(f"BFloat16 Support: {info['supports_bfloat16']}")

# Apply PyTorch optimizations
optimize_pytorch_settings()
```

**What gets optimized:**

1. **cuDNN benchmark mode** - Auto-tunes convolution algorithms
2. **TF32 precision** - Enabled on Ampere GPUs (8.x+) for faster training
3. **Thread count** - Optimal CPU thread allocation
4. **CUDA cache** - Cleared before training

---

## Hardware-Specific Configurations

### RTX 4090 / A6000 (24GB VRAM)

**Config:** `conf/hardware/gpu_4090.yaml`

```yaml
model:
  use_gradient_checkpointing: false  # Plenty of memory

training:
  batch_size: 16                     # Large batches
  num_workers: 8

device:
  mixed_precision: true              # bfloat16
  tf32: true                         # TF32 for matmul

optimization:
  gradient_accumulation_steps: 1     # No need
  compile: true                      # torch.compile enabled
```

**Expected throughput:** ~15-20 samples/sec

**Command:**
```bash
make nli-train-4090
```

---

### RTX 3090 (24GB VRAM)

**Config:** `conf/hardware/gpu_3090.yaml`

```yaml
model:
  use_gradient_checkpointing: false

training:
  batch_size: 12                     # Slightly smaller than 4090
  num_workers: 6

device:
  mixed_precision: true
  tf32: true                         # Ampere architecture

optimization:
  gradient_accumulation_steps: 1
  compile: false                     # May have driver issues
```

**Expected throughput:** ~12-18 samples/sec

**Command:**
```bash
make nli-train-3090
```

---

### Low-Memory GPUs: RTX 3060, 2080 Ti (8-12GB VRAM)

**Config:** `conf/hardware/gpu_low_mem.yaml`

```yaml
model:
  use_gradient_checkpointing: true   # Essential for memory
  freeze_encoder: true               # Must freeze

training:
  batch_size: 4                      # Conservative
  num_workers: 4

device:
  mixed_precision: true              # Critical for memory

optimization:
  gradient_accumulation_steps: 4     # Simulates batch_size=16
  compile: false
```

**Effective batch size:** 4 × 4 = 16 (via gradient accumulation)

**Expected throughput:** ~6-10 samples/sec

**Command:**
```bash
make nli-train-low-mem
```

---

### CPU Training (Not Recommended)

**Config:** `conf/hardware/cpu.yaml`

```yaml
training:
  batch_size: 2                      # Very small
  num_workers: 4

device:
  use_cuda: false
  mixed_precision: false

optimization:
  gradient_accumulation_steps: 8
```

**⚠️ Warning:** CPU training is 50-100x slower than GPU. Only use for testing.

**Expected time:** ~24-48 hours for 5-fold CV

**Command:**
```bash
make nli-train-cpu
```

---

## Optimization Techniques

### 1. Mixed Precision Training (bfloat16)

**Benefits:**
- 2x faster training
- 50% less memory usage
- Native support on Ampere GPUs (RTX 30/40 series)

**Implementation:**
```python
# Automatically enabled in training scripts
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    logits = model(input_ids, attention_mask)
    loss = criterion(logits, labels)
```

**When to use:** Always, unless debugging numerical issues

---

### 2. Gradient Checkpointing

**Benefits:**
- Trades computation for memory
- Enables larger models/batches on limited VRAM
- ~30% slower, but 50% less memory

**When to use:**
- ✅ GPUs with <16GB VRAM
- ✅ Batch size >8 with frozen encoder
- ❌ GPUs with >20GB VRAM (not needed)

**Configuration:**
```yaml
model:
  use_gradient_checkpointing: true
```

---

### 3. Gradient Accumulation

**Purpose:** Simulate larger batch sizes without increasing memory

**Example:**
```yaml
training:
  batch_size: 4
optimization:
  gradient_accumulation_steps: 4
# Effective batch size = 4 × 4 = 16
```

**When to use:**
- Low-memory GPUs that can't fit desired batch size
- Maintaining large effective batch size for stable training

---

### 4. TF32 Precision

**What it is:** TensorFloat-32 - faster float32 with reduced precision

**Benefits:**
- Up to 8x faster matrix multiplications
- No code changes required
- Minimal accuracy impact

**Requirements:**
- Ampere GPU or newer (RTX 30/40 series, A100)
- PyTorch 1.7+

**Enabled automatically on compatible GPUs:**
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

---

### 5. Model Compilation (torch.compile)

**Benefits:**
- 10-30% speedup on PyTorch 2.0+
- Automatic kernel fusion and optimization

**Requirements:**
- PyTorch 2.0+
- CUDA 11.7+
- Stable drivers

**Configuration:**
```yaml
optimization:
  compile: true
  compile_mode: default  # or 'reduce-overhead', 'max-autotune'
```

**⚠️ Note:** May fail on older drivers. Set `compile: false` if training crashes.

---

### 6. DataLoader Optimization

**Optimal settings:**

```yaml
training:
  num_workers: 4-8           # 1 worker per 3GB VRAM
  pin_memory: true           # Faster CPU→GPU transfer
  prefetch_factor: 2         # Prefetch 2 batches
```

**Rules of thumb:**
- **RTX 4090 (24GB):** 8 workers
- **RTX 3090 (24GB):** 6 workers
- **RTX 3060 (12GB):** 4 workers
- **Low-memory (<10GB):** 2-4 workers

---

## Makefile Commands

### Hardware Detection

```bash
# Basic GPU check
make check-gpu

# Detailed hardware info with recommendations
make check-hardware
```

### Hardware-Specific Training

```bash
# Automatic detection (recommends command but doesn't run)
make nli-train-auto

# RTX 4090 optimized
make nli-train-4090

# RTX 3090 optimized
make nli-train-3090

# Low-memory GPUs (8-12GB)
make nli-train-low-mem

# CPU training (not recommended)
make nli-train-cpu
```

### Manual Override

```bash
# Override any config parameter
python src/training/train_nli_5fold.py \
    hardware=gpu_4090 \
    training.batch_size=20 \
    optimization.compile=true
```

---

## Manual Configuration

### Using Hydra Config Groups

All hardware configs are in `conf/hardware/`:

```
conf/hardware/
├── gpu_4090.yaml       # RTX 4090 / A6000
├── gpu_3090.yaml       # RTX 3090
├── gpu_low_mem.yaml    # 8-12GB GPUs
└── cpu.yaml            # CPU-only
```

**Load a hardware config:**

```bash
python src/training/train_nli_5fold.py hardware=gpu_4090
```

**Combine with experiments:**

```bash
python src/training/train_nli_5fold.py \
    experiment=nli_full_5fold \
    hardware=gpu_3090
```

---

### Programmatic Configuration

```python
from src.utils.hardware_optimizer import get_recommended_config

# Get recommended config based on available VRAM
config = get_recommended_config(gpu_memory_gb=24)

print(f"Batch size: {config['batch_size']}")
print(f"Num workers: {config['num_workers']}")
print(f"Gradient checkpointing: {config['gradient_checkpointing']}")
```

---

## Benchmarking

### Measure Training Speed

```python
from src.utils.hardware_optimizer import benchmark_dataloader

# Benchmark your DataLoader
avg_time = benchmark_dataloader(train_loader, num_batches=10)
print(f"Average time per batch: {avg_time * 1000:.2f} ms")
print(f"Batches per second: {1 / avg_time:.2f}")
```

### Track GPU Memory

```python
from src.utils.hardware_optimizer import MemoryTracker

tracker = MemoryTracker()

# Training loop
for epoch in range(num_epochs):
    tracker.reset()

    for batch in dataloader:
        # Training step
        ...
        tracker.update()

    # Report memory usage
    tracker.report()
```

**Output:**
```
GPU Memory Usage:
  Current: 14.32 GB
  Peak: 18.67 GB
  Total: 24.00 GB
  Utilization: 77.8%
```

---

### Performance Comparison

Expected training times for **5-fold CV on ReDSM5 NLI** (~3,000 pairs):

| Hardware | Batch Size | Time/Epoch | Total Time | Speedup |
|----------|------------|------------|------------|---------|
| RTX 4090 | 16 | 2 min | 1.5 hours | 1.0x |
| RTX 3090 | 12 | 3 min | 2.0 hours | 0.75x |
| RTX 3060 | 4 | 8 min | 5.5 hours | 0.27x |
| CPU | 2 | 120 min | 40 hours | 0.016x |

*Assumptions: 20 epochs/fold, frozen encoder, mixed precision*

---

## Troubleshooting

### Out of Memory (OOM) Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size:**
   ```bash
   python src/training/train_nli_5fold.py \
       training.batch_size=4
   ```

2. **Enable gradient checkpointing:**
   ```bash
   python src/training/train_nli_5fold.py \
       model.use_gradient_checkpointing=true
   ```

3. **Use gradient accumulation:**
   ```bash
   python src/training/train_nli_5fold.py \
       training.batch_size=2 \
       optimization.gradient_accumulation_steps=8
   ```

4. **Try low-memory config:**
   ```bash
   make nli-train-low-mem
   ```

5. **Enable memory-efficient mode:**
   ```python
   from src.utils.hardware_optimizer import enable_memory_efficient_mode
   enable_memory_efficient_mode()
   ```

---

### Slow Training

**Symptoms:** Training slower than expected

**Checklist:**

- ✅ **Mixed precision enabled?** Should be `true` for all GPUs
- ✅ **Using GPU?** Check `nvidia-smi` to verify GPU utilization
- ✅ **cuDNN benchmark enabled?** Set `cudnn_benchmark: true`
- ✅ **TF32 enabled?** Should be `true` on Ampere GPUs
- ✅ **Optimal num_workers?** Try 4-8 workers
- ✅ **Disk I/O bottleneck?** Check if data is on SSD

**Debug:**
```bash
# Check GPU utilization
nvidia-smi -l 1

# Benchmark DataLoader
python -c "
from src.data.redsm5_nli_dataset import load_redsm5_nli
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.utils.hardware_optimizer import benchmark_dataloader

tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b')
train_dataset, _, _, _ = load_redsm5_nli('data/redsm5', tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, num_workers=4)

benchmark_dataloader(train_loader, num_batches=50)
"
```

---

### Model Compilation Errors

**Symptoms:**
```
RuntimeError: Compilation failed
```

**Solutions:**

1. **Disable compilation:**
   ```yaml
   optimization:
     compile: false
   ```

2. **Try different mode:**
   ```yaml
   optimization:
     compile: true
     compile_mode: reduce-overhead  # instead of 'default'
   ```

3. **Update PyTorch:**
   ```bash
   pip install --upgrade torch
   ```

---

### Driver/CUDA Issues

**Symptoms:**
- CUDA initialization errors
- TF32/bfloat16 not working

**Solutions:**

1. **Check CUDA version:**
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.version.cuda)"
   ```

2. **Update drivers:** Minimum recommended:
   - CUDA 11.7+ for PyTorch 2.0
   - CUDA 12.1+ for optimal performance

3. **Disable TF32 if problematic:**
   ```yaml
   device:
     tf32: false
   ```

---

## Hardware Optimizer API Reference

### Functions

```python
# Detection
detect_gpu_info() -> Dict[str, Any]
get_recommended_config(gpu_memory_gb: float = None) -> Dict[str, Any]
print_hardware_info() -> None

# Optimization
optimize_pytorch_settings(config: Dict = None) -> None
enable_memory_efficient_mode() -> None
get_optimal_num_workers() -> int

# Model optimization
compile_model(model, mode: str = 'default') -> nn.Module

# Utilities
benchmark_dataloader(dataloader, num_batches: int = 10) -> float

# Memory tracking
class MemoryTracker:
    def reset() -> None
    def update() -> None
    def report() -> None
```

---

## Best Practices

### ✅ Do

1. **Always check hardware first:** `make check-hardware`
2. **Use mixed precision:** Enabled by default
3. **Match batch size to VRAM:** Use recommended configs
4. **Monitor GPU utilization:** Should be >80% during training
5. **Use gradient accumulation** for low-memory GPUs
6. **Freeze encoder** when possible (saves 50% memory)

### ❌ Don't

1. **Don't use CPU** for production training
2. **Don't disable mixed precision** (slower + more memory)
3. **Don't set batch_size too high** (OOM errors)
4. **Don't use too many workers** (CPU bottleneck)
5. **Don't compile models** on old drivers (may crash)

---

## Additional Resources

- **PyTorch Performance Tuning:** https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- **Mixed Precision Training:** https://pytorch.org/docs/stable/amp.html
- **torch.compile Guide:** https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- **Gemma Encoder Paper:** arXiv:2503.02656

---

## Summary

**Quick Commands:**

```bash
# 1. Check your hardware
make check-hardware

# 2. Train with optimal settings
make nli-train-auto        # Detects and recommends
make nli-train-4090        # RTX 4090
make nli-train-3090        # RTX 3090
make nli-train-low-mem     # 8-12GB GPUs

# 3. Override if needed
python src/training/train_nli_5fold.py \
    hardware=gpu_4090 \
    training.batch_size=20
```

**Hardware configs automatically optimize:**
- Batch size
- Num workers
- Gradient checkpointing
- Mixed precision
- TF32
- Model compilation
- Gradient accumulation

**Expected speedup:** 2-3x compared to unoptimized settings!
