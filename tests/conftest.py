"""
Pytest configuration and shared fixtures.

This file contains shared fixtures and configuration for all tests.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture(scope="session")
def set_random_seeds():
    """Set random seeds for reproducibility in tests."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    return 42


@pytest.fixture(scope="function")
def reset_random_state():
    """Reset random state before each test."""
    torch.manual_seed(42)
    yield
    # Cleanup after test if needed


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
