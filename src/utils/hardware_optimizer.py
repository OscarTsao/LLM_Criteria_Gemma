"""
Hardware Optimization Utilities

Automatically detects and configures optimal settings for available hardware.
Based on PyTorch best practices and Gemma Encoder paper recommendations.
"""

import torch
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def detect_gpu_info() -> Dict[str, Any]:
    """Detect GPU information and capabilities."""
    info = {
        'has_gpu': torch.cuda.is_available(),
        'gpu_count': 0,
        'gpu_name': None,
        'gpu_memory_gb': 0,
        'compute_capability': None,
        'supports_bfloat16': False,
        'supports_tf32': False,
    }

    if torch.cuda.is_available():
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9

        # Check compute capability
        major, minor = torch.cuda.get_device_capability(0)
        info['compute_capability'] = f"{major}.{minor}"

        # Ampere (8.x) and newer support bfloat16 natively
        info['supports_bfloat16'] = major >= 8

        # Ampere (8.x) and newer support TF32
        info['supports_tf32'] = major >= 8

    return info


def get_recommended_config(gpu_memory_gb: float = None, gpu_name: str = None) -> Dict[str, Any]:
    """Get recommended configuration based on hardware."""
    if gpu_memory_gb is None:
        gpu_info = detect_gpu_info()
        gpu_memory_gb = gpu_info['gpu_memory_gb']
        gpu_name = gpu_info['gpu_name']

    config = {
        'batch_size': 4,
        'num_workers': 4,
        'gradient_checkpointing': True,
        'gradient_accumulation_steps': 4,
        'mixed_precision': True,
        'compile': False,
    }

    # RTX 4090 / A6000 (24GB+)
    if gpu_memory_gb >= 22:
        config.update({
            'batch_size': 16,
            'num_workers': 8,
            'gradient_checkpointing': False,
            'gradient_accumulation_steps': 1,
            'compile': True,
        })
        logger.info(f"Detected high-end GPU ({gpu_memory_gb:.1f}GB) - using optimal config")

    # RTX 3090 / A5000 (20-24GB)
    elif gpu_memory_gb >= 18:
        config.update({
            'batch_size': 12,
            'num_workers': 6,
            'gradient_checkpointing': False,
            'gradient_accumulation_steps': 1,
            'compile': True,
        })
        logger.info(f"Detected mid-high GPU ({gpu_memory_gb:.1f}GB) - using balanced config")

    # RTX 3080 / A4000 (10-16GB)
    elif gpu_memory_gb >= 10:
        config.update({
            'batch_size': 8,
            'num_workers': 4,
            'gradient_checkpointing': True,
            'gradient_accumulation_steps': 2,
        })
        logger.info(f"Detected mid-range GPU ({gpu_memory_gb:.1f}GB) - using memory-efficient config")

    # RTX 3060 / 2080 Ti (8-12GB)
    elif gpu_memory_gb >= 7:
        config.update({
            'batch_size': 4,
            'num_workers': 4,
            'gradient_checkpointing': True,
            'gradient_accumulation_steps': 4,
        })
        logger.info(f"Detected lower-end GPU ({gpu_memory_gb:.1f}GB) - using conservative config")

    else:
        logger.warning(f"GPU memory ({gpu_memory_gb:.1f}GB) is very low - training may fail")

    return config


def optimize_pytorch_settings(config: Dict[str, Any] = None):
    """Apply PyTorch performance optimizations."""
    gpu_info = detect_gpu_info()

    if not gpu_info['has_gpu']:
        logger.warning("No GPU detected - using CPU (very slow)")
        return

    # Enable cuDNN benchmark for optimal performance
    torch.backends.cudnn.benchmark = True
    logger.info("✓ Enabled cuDNN benchmark mode")

    # Enable TF32 on Ampere GPUs (8.x+)
    if gpu_info['supports_tf32']:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("✓ Enabled TF32 for Ampere GPU")

    # Set optimal number of threads
    num_threads = min(torch.get_num_threads(), 8)
    torch.set_num_threads(num_threads)
    logger.info(f"✓ Set PyTorch threads to {num_threads}")

    # Disable cuDNN deterministic for speed (if not debugging)
    if config and not config.get('deterministic', False):
        torch.backends.cudnn.deterministic = False
        logger.info("✓ Disabled deterministic mode for speed")

    # Empty cache
    torch.cuda.empty_cache()
    logger.info("✓ Cleared CUDA cache")


def get_optimal_num_workers() -> int:
    """Get optimal number of DataLoader workers."""
    try:
        cpu_count = os.cpu_count() or 4
        # Rule of thumb: 4-8 workers per GPU
        optimal = min(cpu_count, 8)
        return optimal
    except:
        return 4


def print_hardware_info():
    """Print detailed hardware information."""
    gpu_info = detect_gpu_info()

    print("\n" + "=" * 60)
    print("HARDWARE INFORMATION")
    print("=" * 60)

    if gpu_info['has_gpu']:
        print(f"GPU: {gpu_info['gpu_name']}")
        print(f"GPU Memory: {gpu_info['gpu_memory_gb']:.2f} GB")
        print(f"GPU Count: {gpu_info['gpu_count']}")
        print(f"Compute Capability: {gpu_info['compute_capability']}")
        print(f"BFloat16 Support: {'Yes' if gpu_info['supports_bfloat16'] else 'No'}")
        print(f"TF32 Support: {'Yes' if gpu_info['supports_tf32'] else 'No'}")

        # Get recommended config
        config = get_recommended_config()
        print("\nRECOMMENDED SETTINGS:")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  Num Workers: {config['num_workers']}")
        print(f"  Gradient Checkpointing: {config['gradient_checkpointing']}")
        print(f"  Gradient Accumulation: {config['gradient_accumulation_steps']} steps")
        print(f"  Mixed Precision: {config['mixed_precision']}")
        print(f"  Torch Compile: {config['compile']}")

        # Effective batch size
        effective_bs = config['batch_size'] * config['gradient_accumulation_steps']
        print(f"\nEffective Batch Size: {effective_bs}")

    else:
        print("GPU: Not available")
        print("CPU Count: {}".format(os.cpu_count()))
        print("\n⚠ Warning: Training on CPU will be extremely slow")
        print("  Recommended: Use a GPU with at least 8GB VRAM")

    print("=" * 60 + "\n")


def enable_memory_efficient_mode():
    """Enable memory-efficient settings for low-memory GPUs."""
    logger.info("Enabling memory-efficient mode...")

    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Set environment variables for memory efficiency
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    # Disable memory caching in transformers
    os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'

    logger.info("✓ Memory-efficient mode enabled")


def benchmark_dataloader(dataloader, num_batches: int = 10):
    """Benchmark DataLoader performance."""
    import time

    print(f"\nBenchmarking DataLoader ({num_batches} batches)...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    times = []
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        start = time.time()

        # Move to device (simulating training)
        if isinstance(batch, dict):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    v.to(device)
        elif isinstance(batch, (list, tuple)):
            for item in batch:
                if isinstance(item, torch.Tensor):
                    item.to(device)

        end = time.time()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    print(f"Average time per batch: {avg_time * 1000:.2f} ms")
    print(f"Estimated batches/second: {1 / avg_time:.2f}")

    return avg_time


class MemoryTracker:
    """Track GPU memory usage during training."""

    def __init__(self):
        self.peak_memory = 0
        self.enabled = torch.cuda.is_available()

    def reset(self):
        """Reset peak memory counter."""
        if self.enabled:
            torch.cuda.reset_peak_memory_stats()
            self.peak_memory = 0

    def update(self):
        """Update peak memory."""
        if self.enabled:
            self.peak_memory = max(
                self.peak_memory,
                torch.cuda.max_memory_allocated() / 1e9
            )

    def report(self):
        """Report memory usage."""
        if self.enabled:
            current = torch.cuda.memory_allocated() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9

            print(f"\nGPU Memory Usage:")
            print(f"  Current: {current:.2f} GB")
            print(f"  Peak: {peak:.2f} GB")
            print(f"  Total: {total:.2f} GB")
            print(f"  Utilization: {peak / total * 100:.1f}%")
        else:
            print("GPU memory tracking not available")


def compile_model(model, mode: str = 'default'):
    """
    Compile model with torch.compile for PyTorch 2.0+.

    Args:
        model: PyTorch model
        mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')

    Returns:
        Compiled model (or original if compilation fails)
    """
    try:
        if hasattr(torch, 'compile'):
            logger.info(f"Compiling model with mode={mode}...")
            compiled = torch.compile(model, mode=mode)
            logger.info("✓ Model compiled successfully")
            return compiled
        else:
            logger.warning("torch.compile not available (PyTorch < 2.0)")
            return model
    except Exception as e:
        logger.warning(f"Model compilation failed: {e}")
        logger.warning("Continuing with uncompiled model")
        return model
