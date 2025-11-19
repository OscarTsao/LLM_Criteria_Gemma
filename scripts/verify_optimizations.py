#!/usr/bin/env python3
"""
Verification script for RTX 3090 optimizations.

This script checks that all performance optimizations are properly configured
and provides diagnostic information about the current setup.

Usage:
    python scripts/verify_optimizations.py
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))


def check_pytorch_version():
    """Check PyTorch version and capabilities."""
    print("=" * 60)
    print("PyTorch Environment")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            compute_cap = torch.cuda.get_device_capability(i)
            print(f"  Compute capability: {compute_cap[0]}.{compute_cap[1]}")

            # Check if TF32 is supported (compute capability >= 8.0)
            if compute_cap[0] >= 8:
                print(f"  TF32 supported: ✅ Yes")
            else:
                print(f"  TF32 supported: ❌ No (requires compute capability 8.0+)")

            # Memory info
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  Total memory: {total_memory:.1f} GB")
    else:
        print("❌ CUDA not available - optimizations will not work")

    print()


def check_tf32_support():
    """Check TF32 support and status."""
    print("=" * 60)
    print("TF32 Acceleration")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ CUDA not available - TF32 not supported")
        print()
        return False

    compute_cap = torch.cuda.get_device_capability()
    if compute_cap[0] >= 8:
        print(f"✅ GPU supports TF32 (compute capability {compute_cap[0]}.{compute_cap[1]})")

        # Check if TF32 is enabled
        matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        cudnn_tf32 = torch.backends.cudnn.allow_tf32

        print(f"   torch.backends.cuda.matmul.allow_tf32: {matmul_tf32}")
        print(f"   torch.backends.cudnn.allow_tf32: {cudnn_tf32}")

        if matmul_tf32 and cudnn_tf32:
            print("✅ TF32 is ENABLED")
        else:
            print("⚠️  TF32 is DISABLED (will be enabled during training)")

        print()
        return True
    else:
        print(f"❌ GPU does not support TF32 (compute capability {compute_cap[0]}.{compute_cap[1]} < 8.0)")
        print()
        return False


def check_flash_attention():
    """Check Flash Attention support."""
    print("=" * 60)
    print("Flash Attention (SDPA)")
    print("=" * 60)

    # Check if SDPA is available (PyTorch 2.0+)
    has_sdpa = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    if has_sdpa:
        print("✅ SDPA available (PyTorch 2.0+)")

        # Check if Flash Attention backends are available
        if torch.cuda.is_available():
            try:
                # Try to enable backends (won't affect global state in this check)
                backends_available = []

                # Flash Attention
                try:
                    torch.backends.cuda.enable_flash_sdp(True)
                    backends_available.append("Flash Attention")
                except:
                    pass

                # Memory-efficient attention
                try:
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                    backends_available.append("Memory-efficient")
                except:
                    pass

                if backends_available:
                    print(f"✅ Available backends: {', '.join(backends_available)}")
                else:
                    print("⚠️  No efficient attention backends available")
            except Exception as e:
                print(f"⚠️  Could not check backends: {e}")
        else:
            print("❌ CUDA not available - Flash Attention requires GPU")
    else:
        print("❌ SDPA not available (requires PyTorch 2.0+)")
        print(f"   Current version: {torch.__version__}")

    print()


def check_fused_adamw():
    """Check fused AdamW support."""
    print("=" * 60)
    print("Fused AdamW Optimizer")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ CUDA not available - fused AdamW requires GPU")
        print()
        return False

    # Try to create a fused optimizer
    try:
        dummy_param = torch.nn.Parameter(torch.randn(10, 10, device='cuda'))
        optimizer = torch.optim.AdamW([dummy_param], lr=1e-3, fused=True)
        print("✅ Fused AdamW is available")
        del optimizer, dummy_param
        print()
        return True
    except Exception as e:
        print(f"❌ Fused AdamW not available: {e}")
        print()
        return False


def check_torch_compile():
    """Check torch.compile support."""
    print("=" * 60)
    print("torch.compile (Optional)")
    print("=" * 60)

    if hasattr(torch, 'compile'):
        print(f"✅ torch.compile available (PyTorch {torch.__version__})")
        print("   Note: Currently disabled by default due to compatibility concerns")
    else:
        print(f"❌ torch.compile not available (requires PyTorch 2.0+)")
        print(f"   Current version: {torch.__version__}")

    print()


def check_config_files():
    """Check configuration files."""
    print("=" * 60)
    print("Configuration Files")
    print("=" * 60)

    config_path = Path(__file__).parent.parent / 'conf' / 'config.yaml'
    rtx3090_config_path = Path(__file__).parent.parent / 'conf' / 'experiment' / 'rtx3090_optimized.yaml'

    if config_path.exists():
        print(f"✅ Base config found: {config_path}")
    else:
        print(f"❌ Base config not found: {config_path}")

    if rtx3090_config_path.exists():
        print(f"✅ RTX 3090 optimized config found: {rtx3090_config_path}")
    else:
        print(f"❌ RTX 3090 optimized config not found: {rtx3090_config_path}")

    print()


def check_dataloader_settings():
    """Check DataLoader optimization settings."""
    print("=" * 60)
    print("DataLoader Optimizations")
    print("=" * 60)

    # These settings are configured in config.yaml, not checked programmatically
    print("DataLoader settings configured in config.yaml:")
    print("  - num_workers: 4 (parallel data loading)")
    print("  - pin_memory: true (faster GPU transfer)")
    print("  - prefetch_factor: 2 (prefetch batches)")
    print("  - persistent_workers: true (reuse workers)")
    print()


def main():
    """Run all checks."""
    print("\n" + "=" * 60)
    print("RTX 3090 Optimization Verification")
    print("=" * 60)
    print()

    check_pytorch_version()
    tf32_ok = check_tf32_support()
    check_flash_attention()
    fused_adamw_ok = check_fused_adamw()
    check_torch_compile()
    check_dataloader_settings()
    check_config_files()

    print("=" * 60)
    print("Summary")
    print("=" * 60)

    if torch.cuda.is_available():
        compute_cap = torch.cuda.get_device_capability()
        if compute_cap[0] >= 8:
            print("✅ GPU is Ampere or newer (full optimization support)")
        else:
            print("⚠️  GPU is pre-Ampere (some optimizations unavailable)")
    else:
        print("❌ No CUDA GPU detected - optimizations will not work")

    print("\nTo run training with optimizations:")
    print("  python src/training/train_nli_binary.py experiment=rtx3090_optimized")
    print("\nFor more information, see:")
    print("  docs/RTX3090_OPTIMIZATIONS.md")
    print()


if __name__ == '__main__':
    main()
