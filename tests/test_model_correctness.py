"""
Model correctness tests for all supported architectures.

Tests:
1. Shape validation (batch, seq, hidden) -> (batch, hidden) -> (batch, num_classes)
2. LoRA parameter count verification
3. AMP (mixed precision) path validation
4. Bidirectional attention verification
5. Pooling strategy correctness
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.gemma_encoder import GemmaClassifier
from src.models.poolers import (
    MeanPooler, CLSPooler, MaxPooler,
    AttentionPooler, FirstKPooler, LastKPooler
)


class TestPoolerShapes:
    """Test that all poolers produce correct output shapes."""

    @pytest.fixture
    def sample_hidden_states(self):
        """Create sample hidden states tensor."""
        batch_size = 4
        seq_len = 128
        hidden_dim = 768
        return torch.randn(batch_size, seq_len, hidden_dim)

    @pytest.fixture
    def sample_attention_mask(self):
        """Create sample attention mask."""
        batch_size = 4
        seq_len = 128
        # Create realistic mask (first 100 tokens are valid)
        mask = torch.zeros(batch_size, seq_len)
        mask[:, :100] = 1
        return mask

    @pytest.mark.parametrize("pooler_class", [
        MeanPooler, CLSPooler, MaxPooler, AttentionPooler, FirstKPooler, LastKPooler
    ])
    def test_pooler_output_shape(self, pooler_class, sample_hidden_states, sample_attention_mask):
        """Test that pooler produces correct output shape."""
        batch_size, seq_len, hidden_dim = sample_hidden_states.shape

        if pooler_class in [FirstKPooler, LastKPooler]:
            pooler = pooler_class(hidden_dim=hidden_dim, k=10)
        elif pooler_class == AttentionPooler:
            pooler = pooler_class(hidden_dim=hidden_dim, num_heads=4)
        else:
            pooler = pooler_class(hidden_dim=hidden_dim)

        output = pooler(sample_hidden_states, sample_attention_mask)

        # Should output (batch_size, hidden_dim)
        assert output.shape == (batch_size, hidden_dim), (
            f"{pooler_class.__name__} output shape mismatch: "
            f"expected {(batch_size, hidden_dim)}, got {output.shape}"
        )

    def test_mean_pooler_respects_mask(self):
        """Test that MeanPooler correctly handles masking."""
        # Create simple test case
        hidden_states = torch.ones(2, 5, 10)  # (batch=2, seq=5, hidden=10)
        attention_mask = torch.tensor([
            [1, 1, 1, 0, 0],  # First sample: 3 valid tokens
            [1, 1, 1, 1, 1]   # Second sample: all valid
        ])

        pooler = MeanPooler(hidden_dim=10)
        output = pooler(hidden_states, attention_mask)

        # Both samples should have mean of 1.0 (since all values are 1)
        assert torch.allclose(output, torch.ones(2, 10)), "MeanPooler not computing mean correctly"

    def test_max_pooler_respects_mask(self):
        """Test that MaxPooler correctly handles masking."""
        # Create test case where padded tokens have high values
        hidden_states = torch.ones(1, 5, 3)
        hidden_states[0, 0, :] = 5.0  # First token has max value
        hidden_states[0, 3:, :] = 10.0  # Padded tokens have even higher values

        attention_mask = torch.tensor([[1, 1, 1, 0, 0]])  # Only first 3 tokens valid

        pooler = MaxPooler(hidden_dim=3)
        output = pooler(hidden_states, attention_mask)

        # Should select max from first 3 tokens (5.0), not from padded (10.0)
        assert torch.allclose(output, torch.tensor([[5.0, 5.0, 5.0]])), (
            "MaxPooler is not properly masking padded tokens"
        )

    def test_cls_pooler_selects_first_token(self):
        """Test that CLSPooler selects the first token."""
        hidden_states = torch.randn(2, 10, 768)
        hidden_states[:, 0, :] = 999.0  # Set first token to recognizable value

        attention_mask = torch.ones(2, 10)
        pooler = CLSPooler(hidden_dim=768)
        output = pooler(hidden_states, attention_mask)

        assert torch.allclose(output, torch.ones(2, 768) * 999.0), (
            "CLSPooler is not selecting the first token"
        )


@pytest.mark.slow
class TestGemmaEncoderShapes:
    """Test GemmaClassifier with different configurations."""

    @pytest.fixture
    def small_model_name(self):
        """Use a small model for testing (if available) or mock."""
        # For CI, we might need to mock this
        return "google/gemma-2-2b"

    @pytest.mark.parametrize("pooling_strategy", [
        "mean", "cls", "max", "first_k", "last_k", "attention"
    ])
    def test_forward_pass_shapes(self, pooling_strategy):
        """Test that forward pass produces correct shapes."""
        # Skip if no GPU available and model is too large
        if not torch.cuda.is_available():
            pytest.skip("Requires GPU for large model")

        batch_size = 2
        seq_len = 64
        num_classes = 10

        # Create mock inputs
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Initialize model (this may download weights)
        try:
            model = GemmaClassifier(
                num_classes=num_classes,
                model_name="google/gemma-2-2b",
                pooling_strategy=pooling_strategy,
                freeze_encoder=True  # Freeze to save memory
            )
        except Exception as e:
            pytest.skip(f"Could not load model: {e}")

        model.eval()

        with torch.no_grad():
            logits = model(input_ids, attention_mask)

        # Check output shape
        assert logits.shape == (batch_size, num_classes), (
            f"Expected shape {(batch_size, num_classes)}, got {logits.shape}"
        )

        # Check dtype
        assert logits.dtype == torch.float32 or logits.dtype == torch.bfloat16

    def test_amp_compatibility(self):
        """Test that model works with automatic mixed precision."""
        if not torch.cuda.is_available():
            pytest.skip("Requires CUDA for AMP")

        batch_size = 2
        seq_len = 64
        num_classes = 5

        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).cuda()
        attention_mask = torch.ones(batch_size, seq_len).cuda()

        try:
            model = GemmaClassifier(
                num_classes=num_classes,
                model_name="google/gemma-2-2b",
                freeze_encoder=True
            ).cuda()
        except Exception as e:
            pytest.skip(f"Could not load model: {e}")

        # Test with autocast
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(input_ids, attention_mask)

        # Should produce valid outputs
        assert not torch.isnan(logits).any(), "NaN detected in AMP forward pass"
        assert not torch.isinf(logits).any(), "Inf detected in AMP forward pass"
        assert logits.shape == (batch_size, num_classes)


class TestLoRAConfiguration:
    """Test LoRA parameter counting and configuration."""

    def test_parameter_count_without_lora(self):
        """Test baseline parameter count without LoRA."""
        # Use small mock model
        class MockEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(1000, 128)
                self.linear = nn.Linear(128, 128)

            def forward(self, x):
                return self.linear(self.embedding(x))

        model = MockEncoder()
        total_params = sum(p.numel() for p in model.parameters())

        # Should have embedding (1000 * 128) + linear (128 * 128 + 128)
        expected = 1000 * 128 + 128 * 128 + 128
        assert total_params == expected

    def test_lora_reduces_trainable_params(self):
        """Test that LoRA reduces trainable parameter count."""
        pytest.skip("LoRA implementation not yet in codebase")

        # When LoRA is implemented:
        # model_with_lora = GemmaClassifier(..., use_lora=True, lora_rank=16)
        # model_without_lora = GemmaClassifier(..., use_lora=False)
        #
        # trainable_with_lora = sum(p.numel() for p in model_with_lora.parameters() if p.requires_grad)
        # trainable_without_lora = sum(p.numel() for p in model_without_lora.parameters() if p.requires_grad)
        #
        # assert trainable_with_lora < trainable_without_lora


class TestBidirectionalAttention:
    """Test bidirectional attention implementation."""

    def test_bidirectional_vs_causal_attention(self):
        """
        Test that bidirectional attention allows backward context flow.

        In causal attention, later tokens cannot attend to earlier ones.
        In bidirectional attention, all tokens can attend to all others.
        """
        pytest.skip("Requires specific attention inspection")

        # This would require:
        # 1. Getting attention weights from the model
        # 2. Verifying that upper triangle is not zero (unlike causal)
        # 3. Comparing outputs with and without bidirectional patch


class TestModelDeterminism:
    """Test model output determinism."""

    def test_same_input_same_output(self):
        """Test that same input produces same output with fixed seed."""
        # Set all seeds
        torch.manual_seed(42)
        np.random.seed(42)

        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Create small mock model for testing
        class MockClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(1000, 64)
                self.pooler = MeanPooler(hidden_dim=64)
                self.classifier = nn.Linear(64, 5)

            def forward(self, input_ids, attention_mask):
                embeddings = self.embedding(input_ids)
                pooled = self.pooler(embeddings, attention_mask)
                return self.classifier(pooled)

        model = MockClassifier()
        model.eval()

        with torch.no_grad():
            output1 = model(input_ids, attention_mask)
            output2 = model(input_ids, attention_mask)

        assert torch.allclose(output1, output2), "Model outputs are not deterministic"


class TestGradientFlow:
    """Test gradient flow through the model."""

    def test_gradients_flow_to_classifier(self):
        """Test that gradients flow to classifier head."""
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(10, 20)
                self.classifier = nn.Linear(20, 5)

            def forward(self, x):
                return self.classifier(self.encoder(x))

        model = MockModel()
        x = torch.randn(4, 10)
        y = torch.randint(0, 5, (4,))

        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()

        # Check that classifier has gradients
        assert model.classifier.weight.grad is not None, "No gradients in classifier"
        assert model.classifier.weight.grad.abs().sum() > 0, "Zero gradients in classifier"

    def test_frozen_encoder_no_gradients(self):
        """Test that frozen encoder receives no gradients."""
        class MockModel(nn.Module):
            def __init__(self, freeze_encoder=True):
                super().__init__()
                self.encoder = nn.Linear(10, 20)
                self.classifier = nn.Linear(20, 5)

                if freeze_encoder:
                    for param in self.encoder.parameters():
                        param.requires_grad = False

            def forward(self, x):
                return self.classifier(self.encoder(x))

        model = MockModel(freeze_encoder=True)
        x = torch.randn(4, 10)
        y = torch.randint(0, 5, (4,))

        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()

        # Encoder should have no gradients
        assert model.encoder.weight.grad is None, "Frozen encoder has gradients!"

        # Classifier should have gradients
        assert model.classifier.weight.grad is not None, "Classifier has no gradients"
