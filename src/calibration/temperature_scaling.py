"""
Temperature Scaling for probability calibration.

Reference: Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TemperatureScaling(nn.Module):
    """
    Temperature scaling calibration for multi-class classification.

    Learns a single temperature parameter T to scale logits:
        p_calibrated = softmax(logits / T)

    Args:
        num_classes: Number of classes
        device: Device to run optimization on
    """

    def __init__(self, num_classes: int, device: str = 'cpu'):
        super().__init__()
        self.num_classes = num_classes
        self.device = device

        # Initialize temperature to 1.0 (identity)
        self.temperature = nn.Parameter(torch.ones(1, device=device))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.

        Args:
            logits: (batch, num_classes) logit tensor

        Returns:
            Calibrated probabilities (batch, num_classes)
        """
        # Guard against invalid temperature
        temp = torch.clamp(self.temperature, min=0.01, max=100.0)
        return F.softmax(logits / temp, dim=1)

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 50,
        batch_size: int = 256,
        verbose: bool = True
    ) -> float:
        """
        Fit temperature parameter to minimize NLL on validation set.

        Args:
            logits: (N, num_classes) array of pre-softmax logits
            labels: (N,) array of integer labels
            lr: Learning rate
            max_iter: Maximum iterations
            batch_size: Batch size for optimization
            verbose: Print progress

        Returns:
            Final NLL loss
        """
        # Convert to tensors
        logits_t = torch.from_numpy(logits).float().to(self.device)
        labels_t = torch.from_numpy(labels).long().to(self.device)

        # Create dataloader
        dataset = TensorDataset(logits_t, labels_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Optimizer
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        criterion = nn.CrossEntropyLoss()

        def eval_loss():
            loss = 0.0
            for batch_logits, batch_labels in loader:
                scaled_logits = batch_logits / torch.clamp(self.temperature, min=0.01, max=100.0)
                loss += criterion(scaled_logits, batch_labels).item()
            return loss / len(loader)

        # Optimization closure
        def closure():
            optimizer.zero_grad()
            total_loss = 0.0
            for batch_logits, batch_labels in loader:
                scaled_logits = batch_logits / torch.clamp(self.temperature, min=0.01, max=100.0)
                loss = criterion(scaled_logits, batch_labels)
                total_loss += loss
            total_loss.backward()
            return total_loss

        # Initial loss
        initial_nll = eval_loss()

        # Optimize
        optimizer.step(closure)

        # Final loss
        final_nll = eval_loss()

        if verbose:
            logger.info(f"Temperature Scaling: T={self.temperature.item():.4f}, "
                       f"NLL: {initial_nll:.4f} -> {final_nll:.4f}")

        return final_nll

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply learned temperature to new logits.

        Args:
            logits: (N, num_classes) array of logits

        Returns:
            Calibrated probabilities (N, num_classes)
        """
        with torch.no_grad():
            logits_t = torch.from_numpy(logits).float().to(self.device)
            probs = self(logits_t)
            return probs.cpu().numpy()

    def get_temperature(self) -> float:
        """Return the learned temperature value."""
        return self.temperature.item()


def validate_monotonicity(
    logits: np.ndarray,
    temperatures: np.ndarray = np.linspace(0.1, 10.0, 100)
) -> bool:
    """
    Test that probabilities are monotonic in temperature.

    For a fixed logit vector, as T increases, the probability distribution
    should become more uniform (entropy increases).

    Args:
        logits: (num_classes,) single logit vector
        temperatures: Temperature values to test

    Returns:
        True if entropy is monotonically increasing with temperature
    """
    entropies = []
    logits_t = torch.from_numpy(logits).float()

    for temp in temperatures:
        probs = F.softmax(logits_t / temp, dim=0)
        entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
        entropies.append(entropy)

    # Check monotonicity
    return all(entropies[i] <= entropies[i+1] for i in range(len(entropies)-1))
