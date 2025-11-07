"""
Calibration methods for probability outputs.

Implements temperature scaling and isotonic regression for calibrating
multi-class classification probabilities.
"""

from .temperature_scaling import TemperatureScaling
from .isotonic_calibration import IsotonicCalibration
from .metrics import expected_calibration_error, reliability_diagram

__all__ = [
    'TemperatureScaling',
    'IsotonicCalibration',
    'expected_calibration_error',
    'reliability_diagram',
]
