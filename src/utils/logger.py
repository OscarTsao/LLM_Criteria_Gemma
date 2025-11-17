"""
Logging configuration for the project.

Provides consistent logging across all modules with proper formatting
and file/console output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


# ANSI color codes for console output
class LogColors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    GRAY = '\033[90m'


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    LEVEL_COLORS = {
        'DEBUG': LogColors.GRAY,
        'INFO': LogColors.GREEN,
        'WARNING': LogColors.YELLOW,
        'ERROR': LogColors.RED,
        'CRITICAL': LogColors.RED,
    }

    def format(self, record):
        """Format log record with colors."""
        if sys.stdout.isatty():  # Only use colors if outputting to terminal
            levelname = record.levelname
            if levelname in self.LEVEL_COLORS:
                record.levelname = (
                    f"{self.LEVEL_COLORS[levelname]}{levelname}{LogColors.RESET}"
                )
        return super().format(record)


def setup_logger(
    name: str = 'gemma_encoder',
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True,
    file_level: int = logging.DEBUG,
    console_level: int = logging.INFO,
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Args:
        name: Logger name
        level: Default logging level
        log_file: Optional path to log file
        console: Whether to output to console
        file_level: Logging level for file handler
        console_level: Logging level for console handler

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler with colors
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)

        console_format = ColoredFormatter(
            fmt='%(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    # File handler (detailed)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(file_level)

        file_format = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a default one.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger has no handlers, set up with defaults
    if not logger.handlers:
        return setup_logger(name)

    return logger


def log_experiment_config(logger: logging.Logger, config: dict):
    """
    Log experiment configuration in a formatted way.

    Args:
        logger: Logger instance
        config: Configuration dictionary
    """
    logger.info("=" * 60)
    logger.info("Experiment Configuration")
    logger.info("=" * 60)

    def log_dict(d, prefix=""):
        """Recursively log dictionary."""
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info(f"{prefix}{key}:")
                log_dict(value, prefix=prefix + "  ")
            else:
                logger.info(f"{prefix}{key}: {value}")

    log_dict(config)
    logger.info("=" * 60)


def log_metrics(logger: logging.Logger, metrics: dict, prefix: str = ""):
    """
    Log metrics in a formatted way.

    Args:
        logger: Logger instance
        metrics: Dictionary of metric names and values
        prefix: Optional prefix for metric names
    """
    for name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{prefix}{name}: {value:.4f}")
        else:
            logger.info(f"{prefix}{name}: {value}")


def log_training_summary(
    logger: logging.Logger,
    fold: int,
    epoch: int,
    train_loss: float,
    val_metrics: dict,
    is_best: bool = False,
):
    """
    Log training summary for an epoch.

    Args:
        logger: Logger instance
        fold: Fold number (for cross-validation)
        epoch: Epoch number
        train_loss: Training loss
        val_metrics: Validation metrics dictionary
        is_best: Whether this is the best model so far
    """
    logger.info(f"Fold {fold} | Epoch {epoch}")
    logger.info(f"  Train Loss: {train_loss:.4f}")

    for metric_name, metric_value in val_metrics.items():
        if isinstance(metric_value, float):
            logger.info(f"  Val {metric_name.capitalize()}: {metric_value:.4f}")

    if is_best:
        logger.info("  ✓ Best model saved!")


# Example usage and testing
if __name__ == '__main__':
    # Example 1: Console-only logger
    logger1 = setup_logger('test_console', level=logging.INFO)
    logger1.debug("This is a debug message")
    logger1.info("This is an info message")
    logger1.warning("This is a warning message")
    logger1.error("This is an error message")

    # Example 2: Logger with file output
    logger2 = setup_logger(
        'test_file',
        level=logging.DEBUG,
        log_file=Path('/tmp/test.log'),
    )
    logger2.info("This goes to both console and file")

    # Example 3: Log configuration
    config = {
        'model': {'name': 'gemma-3-4b-it', 'pooling': 'mean'},
        'training': {'lr': 2e-5, 'epochs': 10},
    }
    log_experiment_config(logger2, config)

    # Example 4: Log metrics
    metrics = {'accuracy': 0.85, 'f1': 0.82, 'loss': 0.45}
    log_metrics(logger2, metrics, prefix="Validation ")
