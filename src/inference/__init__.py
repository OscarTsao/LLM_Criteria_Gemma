"""Inference module for NLI binary classification."""

from .predict_nli import predict_single, load_model

__all__ = ['predict_single', 'load_model']
