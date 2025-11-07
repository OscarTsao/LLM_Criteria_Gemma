"""
Unit tests for pooling strategies.

Tests all pooler implementations to ensure correct behavior.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.poolers import (
    MeanPooler,
    CLSPooler,
    MaxPooler,
    FirstKPooler,
    LastKPooler,
    AttentionPooler,
    get_pooler,
)


class TestMeanPooler:
    """Tests for MeanPooler."""

    def test_mean_pooler_no_mask(self):
        """Test mean pooling without attention mask."""
        pooler = MeanPooler()
        hidden_states = torch.randn(2, 5, 10)  # batch=2, seq=5, hidden=10
        result = pooler(hidden_states)

        assert result.shape == (2, 10)
        # Should be equal to mean over sequence dimension
        expected = hidden_states.mean(dim=1)
        torch.testing.assert_close(result, expected)

    def test_mean_pooler_with_mask(self):
        """Test mean pooling with attention mask."""
        pooler = MeanPooler()
        hidden_states = torch.randn(2, 5, 10)
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.float)

        result = pooler(hidden_states, attention_mask)
        assert result.shape == (2, 10)

        # Verify first sample: mean of first 3 tokens
        expected_0 = hidden_states[0, :3, :].mean(dim=0)
        torch.testing.assert_close(result[0], expected_0, rtol=1e-4, atol=1e-4)

        # Verify second sample: mean of first 4 tokens
        expected_1 = hidden_states[1, :4, :].mean(dim=0)
        torch.testing.assert_close(result[1], expected_1, rtol=1e-4, atol=1e-4)

    def test_mean_pooler_all_masked(self):
        """Test mean pooling when all tokens are masked (edge case)."""
        pooler = MeanPooler()
        hidden_states = torch.randn(1, 5, 10)
        attention_mask = torch.zeros(1, 5)  # All masked

        result = pooler(hidden_states, attention_mask)
        assert result.shape == (1, 10)
        # Should handle division by zero gracefully (clamped to 1e-9)
        assert torch.isfinite(result).all()


class TestCLSPooler:
    """Tests for CLSPooler."""

    def test_cls_pooler(self):
        """Test CLS token pooling."""
        pooler = CLSPooler()
        hidden_states = torch.randn(2, 5, 10)
        result = pooler(hidden_states)

        assert result.shape == (2, 10)
        # Should be equal to first token
        torch.testing.assert_close(result, hidden_states[:, 0, :])

    def test_cls_pooler_ignores_mask(self):
        """Test that CLS pooler ignores attention mask."""
        pooler = CLSPooler()
        hidden_states = torch.randn(2, 5, 10)
        attention_mask = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]])

        result = pooler(hidden_states, attention_mask)
        # Should still return first token even if masked
        torch.testing.assert_close(result, hidden_states[:, 0, :])


class TestMaxPooler:
    """Tests for MaxPooler."""

    def test_max_pooler_no_mask(self):
        """Test max pooling without attention mask."""
        pooler = MaxPooler()
        hidden_states = torch.randn(2, 5, 10)
        result = pooler(hidden_states)

        assert result.shape == (2, 10)
        expected = hidden_states.max(dim=1)[0]
        torch.testing.assert_close(result, expected)

    def test_max_pooler_with_mask(self):
        """Test max pooling with attention mask."""
        pooler = MaxPooler()
        # Create controlled hidden states
        hidden_states = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],  # Second sample
        ])
        attention_mask = torch.tensor([[1, 1, 0, 0]])  # Only first 2 tokens valid

        result = pooler(hidden_states, attention_mask)
        assert result.shape == (1, 2)

        # Max should be from first 2 tokens only
        expected = torch.tensor([[3.0, 4.0]])
        torch.testing.assert_close(result, expected)

    def test_max_pooler_memory_efficiency(self):
        """Test that max pooler doesn't modify original tensor."""
        pooler = MaxPooler()
        hidden_states = torch.randn(2, 5, 10)
        original_copy = hidden_states.clone()
        attention_mask = torch.ones(2, 5)

        result = pooler(hidden_states, attention_mask)

        # Original tensor should be unchanged
        torch.testing.assert_close(hidden_states, original_copy)


class TestFirstKPooler:
    """Tests for FirstKPooler."""

    def test_first_k_pooler(self):
        """Test first-K token pooling."""
        pooler = FirstKPooler(k=2)
        hidden_states = torch.randn(2, 5, 10)
        result = pooler(hidden_states)

        assert result.shape == (2, 10)
        # Should be mean of first 2 tokens
        expected = hidden_states[:, :2, :].mean(dim=1)
        torch.testing.assert_close(result, expected)

    def test_first_k_pooler_k1(self):
        """Test first-K pooler with k=1."""
        pooler = FirstKPooler(k=1)
        hidden_states = torch.randn(2, 5, 10)
        result = pooler(hidden_states)

        assert result.shape == (2, 10)
        torch.testing.assert_close(result, hidden_states[:, 0, :])


class TestLastKPooler:
    """Tests for LastKPooler."""

    def test_last_k_pooler(self):
        """Test last-K token pooling."""
        pooler = LastKPooler(k=2)
        hidden_states = torch.randn(2, 5, 10)
        result = pooler(hidden_states)

        assert result.shape == (2, 10)
        # Should be mean of last 2 tokens
        expected = hidden_states[:, -2:, :].mean(dim=1)
        torch.testing.assert_close(result, expected)


class TestAttentionPooler:
    """Tests for AttentionPooler."""

    def test_attention_pooler_initialization(self):
        """Test AttentionPooler initialization."""
        hidden_dim = 128
        pooler = AttentionPooler(hidden_dim)

        assert pooler.query.shape == (1, 1, hidden_dim)
        assert pooler.key_proj.in_features == hidden_dim
        assert pooler.key_proj.out_features == hidden_dim
        assert pooler.value_proj.in_features == hidden_dim
        assert pooler.value_proj.out_features == hidden_dim

    def test_attention_pooler_forward(self):
        """Test AttentionPooler forward pass."""
        hidden_dim = 128
        pooler = AttentionPooler(hidden_dim)
        hidden_states = torch.randn(2, 5, hidden_dim)

        result = pooler(hidden_states)
        assert result.shape == (2, hidden_dim)

    def test_attention_pooler_with_mask(self):
        """Test AttentionPooler with attention mask."""
        hidden_dim = 128
        pooler = AttentionPooler(hidden_dim)
        hidden_states = torch.randn(2, 5, hidden_dim)
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])

        result = pooler(hidden_states, attention_mask)
        assert result.shape == (2, hidden_dim)
        assert torch.isfinite(result).all()

    def test_attention_pooler_is_learnable(self):
        """Test that AttentionPooler has learnable parameters."""
        pooler = AttentionPooler(128)
        params = list(pooler.parameters())

        assert len(params) > 0
        assert all(p.requires_grad for p in params)


class TestPoolerRegistry:
    """Tests for pooler registry and factory function."""

    def test_get_pooler_mean(self):
        """Test getting mean pooler from registry."""
        pooler = get_pooler('mean')
        assert isinstance(pooler, MeanPooler)

    def test_get_pooler_cls(self):
        """Test getting CLS pooler from registry."""
        pooler = get_pooler('cls')
        assert isinstance(pooler, CLSPooler)

    def test_get_pooler_max(self):
        """Test getting max pooler from registry."""
        pooler = get_pooler('max')
        assert isinstance(pooler, MaxPooler)

    def test_get_pooler_first_k(self):
        """Test getting first-K pooler from registry."""
        pooler = get_pooler('first_k', k=3)
        assert isinstance(pooler, FirstKPooler)
        assert pooler.k == 3

    def test_get_pooler_last_k(self):
        """Test getting last-K pooler from registry."""
        pooler = get_pooler('last_k', k=3)
        assert isinstance(pooler, LastKPooler)
        assert pooler.k == 3

    def test_get_pooler_attention(self):
        """Test getting attention pooler from registry."""
        pooler = get_pooler('attention', hidden_dim=256)
        assert isinstance(pooler, AttentionPooler)

    def test_get_pooler_attention_requires_hidden_dim(self):
        """Test that attention pooler requires hidden_dim parameter."""
        with pytest.raises(ValueError, match="AttentionPooler requires 'hidden_dim'"):
            get_pooler('attention')

    def test_get_pooler_invalid_name(self):
        """Test error for invalid pooler name."""
        with pytest.raises(ValueError, match="Unknown pooler"):
            get_pooler('invalid_pooler')


class TestPoolerCompatibility:
    """Tests for pooler compatibility and edge cases."""

    def test_all_poolers_same_output_shape(self):
        """Test that all poolers produce same output shape."""
        batch_size, seq_len, hidden_dim = 4, 10, 128
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        attention_mask = torch.ones(batch_size, seq_len)

        poolers = [
            MeanPooler(),
            CLSPooler(),
            MaxPooler(),
            FirstKPooler(k=1),
            LastKPooler(k=1),
            AttentionPooler(hidden_dim),
        ]

        for pooler in poolers:
            result = pooler(hidden_states, attention_mask)
            assert result.shape == (batch_size, hidden_dim), f"Failed for {type(pooler).__name__}"

    def test_poolers_handle_different_seq_lengths(self):
        """Test poolers with varying sequence lengths."""
        batch_size, hidden_dim = 2, 64

        for seq_len in [1, 5, 10, 50]:
            hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
            attention_mask = torch.ones(batch_size, seq_len)

            poolers = [
                MeanPooler(),
                CLSPooler(),
                MaxPooler(),
            ]

            for pooler in poolers:
                result = pooler(hidden_states, attention_mask)
                assert result.shape == (batch_size, hidden_dim)

    def test_poolers_handle_large_tensors(self):
        """Test poolers with larger tensors (memory test)."""
        batch_size, seq_len, hidden_dim = 16, 512, 768
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        attention_mask = torch.ones(batch_size, seq_len)

        poolers = [
            MeanPooler(),
            MaxPooler(),
        ]

        for pooler in poolers:
            result = pooler(hidden_states, attention_mask)
            assert result.shape == (batch_size, hidden_dim)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
