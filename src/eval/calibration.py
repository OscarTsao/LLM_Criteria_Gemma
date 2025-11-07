"""Calibration utilities for multi-label predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression


@dataclass
class CalibrationResult:
    method: str
    parameters: Dict[str, Any]
    probabilities: np.ndarray
    logits: Optional[np.ndarray] = None


class TemperatureCalibrator:
    """Single-parameter temperature scaling."""

    def __init__(self, init_temperature: float = 1.0, max_iter: int = 1000, lr: float = 0.01) -> None:
        self.temperature = float(init_temperature)
        self.max_iter = max_iter
        self.lr = lr

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        logits_tensor = torch.from_numpy(logits).float()
        labels_tensor = torch.from_numpy(labels).float()
        log_temp = torch.nn.Parameter(torch.log(torch.tensor(self.temperature)))
        optimizer = torch.optim.LBFGS([log_temp], lr=self.lr, max_iter=self.max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            temperature = torch.exp(log_temp)
            scaled = logits_tensor / temperature
            loss = F.binary_cross_entropy_with_logits(scaled, labels_tensor)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(torch.exp(log_temp).detach().cpu().item())
        return self.temperature

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return logits / self.temperature


class IsotonicCalibrator:
    """Per-class isotonic regression."""

    def __init__(self, out_of_bounds: str = "clip") -> None:
        self.out_of_bounds = out_of_bounds
        self.models: list[Optional[IsotonicRegression]] = []

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> None:
        self.models = []
        num_classes = probabilities.shape[1]
        for idx in range(num_classes):
            model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds=self.out_of_bounds)
            try:
                model.fit(probabilities[:, idx], labels[:, idx])
            except ValueError:
                model = None
            self.models.append(model)

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        if not self.models:
            return probabilities
        calibrated = np.zeros_like(probabilities)
        for idx, model in enumerate(self.models):
            if model is None:
                calibrated[:, idx] = probabilities[:, idx]
            else:
                calibrated[:, idx] = model.transform(probabilities[:, idx])
        return np.clip(calibrated, 0.0, 1.0)


def calibrate(
    logits: Optional[np.ndarray],
    probabilities: np.ndarray,
    labels: np.ndarray,
    method: str = "none",
    temperature_init: float = 1.0,
    isotonic_out_of_bounds: str = "clip",
) -> CalibrationResult:
    method = method.lower()
    if method == "none":
        return CalibrationResult(method="none", parameters={}, probabilities=probabilities, logits=logits)
    if method == "temp":
        if logits is None:
            raise ValueError("Temperature scaling requires logits.")
        calibrator = TemperatureCalibrator(init_temperature=temperature_init)
        calibrator.fit(logits, labels)
        calibrated_logits = calibrator.transform(logits)
        calibrated_probs = 1.0 / (1.0 + np.exp(-calibrated_logits))
        return CalibrationResult(
            method="temp",
            parameters={"temperature": calibrator.temperature},
            probabilities=calibrated_probs,
            logits=calibrated_logits,
        )
    if method == "isotonic":
        calibrator = IsotonicCalibrator(out_of_bounds=isotonic_out_of_bounds)
        calibrator.fit(probabilities, labels)
        calibrated_probs = calibrator.transform(probabilities)
        return CalibrationResult(
            method="isotonic",
            parameters={"out_of_bounds": isotonic_out_of_bounds},
            probabilities=calibrated_probs,
            logits=logits,
        )
    raise ValueError(f"Unknown calibration method '{method}'.")
