import os
import sys
from pathlib import Path
import types
from typing import Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch  # noqa: E402
import numpy as np  # noqa: E402
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")

    class PreTrainedTokenizerBase:
        pass

    class PretrainedConfig:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError("AutoTokenizer stub cannot load pretrained models during tests.")

    class AutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError("AutoModel stub cannot load pretrained models during tests.")

    class AutoModelForCausalLM(AutoModel):
        pass

    class AutoConfig:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError("AutoConfig stub cannot load pretrained configs during tests.")

    transformers_stub.AutoTokenizer = AutoTokenizer
    transformers_stub.AutoModel = AutoModel
    transformers_stub.AutoModelForCausalLM = AutoModelForCausalLM
    transformers_stub.AutoConfig = AutoConfig
    transformers_stub.PretrainedConfig = PretrainedConfig

    token_utils = types.ModuleType("transformers.tokenization_utils_base")
    token_utils.PreTrainedTokenizerBase = PreTrainedTokenizerBase
    transformers_stub.tokenization_utils_base = token_utils

    sys.modules["transformers"] = transformers_stub
    sys.modules["transformers.tokenization_utils_base"] = token_utils

if "sklearn" not in sys.modules:
    sklearn_stub = types.ModuleType("sklearn")
    isotonic_module = types.ModuleType("sklearn.isotonic")
    metrics_module = types.ModuleType("sklearn.metrics")
    model_selection_module = types.ModuleType("sklearn.model_selection")

    class IsotonicRegression:
        def __init__(self, y_min=0.0, y_max=1.0, out_of_bounds="clip"):
            self.y_min = y_min
            self.y_max = y_max
            self.out_of_bounds = out_of_bounds
            self._fitted = False

        def fit(self, x, y):
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            order = np.argsort(x)
            self.x_ = x[order]
            self.y_ = y[order]
            self._fitted = True
            return self

        def transform(self, x):
            if not self._fitted:
                raise RuntimeError("IsotonicRegression stub not fitted.")
            x = np.asarray(x, dtype=np.float64)
            values = np.interp(x, self.x_, self.y_)
            return np.clip(values, self.y_min, self.y_max)

    def _binary_f1(y_true, y_pred, zero_division=0.0):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        tp = np.logical_and(y_true == 1, y_pred == 1).sum()
        fp = np.logical_and(y_true == 0, y_pred == 1).sum()
        fn = np.logical_and(y_true == 1, y_pred == 0).sum()
        precision = tp / (tp + fp) if (tp + fp) else zero_division
        recall = tp / (tp + fn) if (tp + fn) else zero_division
        denom = precision + recall
        if denom == 0:
            return float(zero_division)
        return float(2 * precision * recall / denom)

    def f1_score(y_true, y_pred, average="binary", zero_division=0):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.ndim == 1 or average == "binary":
            return _binary_f1(y_true, y_pred, zero_division=zero_division)
        scores = [
            _binary_f1(y_true[:, idx], y_pred[:, idx], zero_division=zero_division)
            for idx in range(y_true.shape[1])
        ]
        if average == "macro":
            return float(np.mean(scores))
        return np.array(scores, dtype=np.float64)

    def average_precision_score(y_true, y_score):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        order = np.argsort(-y_score)
        y_true = y_true[order]
        y_score = y_score[order]
        total_pos = y_true.sum()
        if total_pos == 0:
            return 0.0
        tp = np.cumsum(y_true)
        precision = tp / (np.arange(len(y_true)) + 1)
        recall = tp / total_pos
        recall = np.concatenate(([0.0], recall))
        precision = np.concatenate(([precision[0]], precision))
        return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))

    def precision_recall_curve(y_true, y_score):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        order = np.argsort(-y_score)
        y_true = y_true[order]
        y_score = y_score[order]
        tp = np.cumsum(y_true)
        fp = np.cumsum(1 - y_true)
        total_pos = y_true.sum()
        if total_pos == 0:
            recall = np.zeros(len(y_true))
        else:
            recall = tp / total_pos
        precision = tp / (tp + fp + 1e-12)
        precision = np.concatenate(([1.0], precision))
        recall = np.concatenate(([0.0], recall))
        thresholds = y_score[:-1]
        return precision, recall, thresholds

    isotonic_module.IsotonicRegression = IsotonicRegression

    class _BaseFold:
        def __init__(self, n_splits=5, shuffle=False, random_state=None) -> None:
            self.n_splits = n_splits
            self.shuffle = shuffle
            self.random_state = random_state

        def _get_indices(self, n_samples: int):
            indices = np.arange(n_samples)
            if self.shuffle:
                rng = np.random.default_rng(self.random_state)
                rng.shuffle(indices)
            fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
            fold_sizes[: n_samples % self.n_splits] += 1
            current = 0
            for fold_size in fold_sizes:
                start, stop = current, current + fold_size
                val_idx = indices[start:stop]
                train_idx = np.concatenate((indices[:start], indices[stop:]))
                yield train_idx, val_idx
                current = stop

    class KFold(_BaseFold):
        def split(self, X, y=None):
            n_samples = len(X)
            for train_idx, val_idx in self._get_indices(n_samples):
                yield train_idx, val_idx

    class StratifiedKFold(_BaseFold):
        def split(self, X, y):
            y = np.asarray(y)
            uniques, inverse = np.unique(y, return_inverse=True)
            bins: Dict[int, List[int]] = {int(cls): [] for cls in range(len(uniques))}
            for idx, cls in enumerate(inverse):
                bins[int(cls)].append(idx)
            fold_bins = [[] for _ in range(self.n_splits)]
            for indices in bins.values():
                indices = np.asarray(indices)
                if self.shuffle:
                    rng = np.random.default_rng(self.random_state)
                    rng.shuffle(indices)
                fold_sizes = np.full(self.n_splits, len(indices) // self.n_splits, dtype=int)
                fold_sizes[: len(indices) % self.n_splits] += 1
                current = 0
                for fold_idx, fold_size in enumerate(fold_sizes):
                    start, stop = current, current + fold_size
                    fold_bins[fold_idx].extend(indices[start:stop])
                    current = stop
            for fold_idx in range(self.n_splits):
                val_idx = np.array(sorted(fold_bins[fold_idx]))
                train_idx = np.array(sorted(set(range(len(X))) - set(val_idx)))
                yield train_idx, val_idx

    model_selection_module.KFold = KFold
    model_selection_module.StratifiedKFold = StratifiedKFold
    metrics_module.f1_score = f1_score
    metrics_module.average_precision_score = average_precision_score
    metrics_module.precision_recall_curve = precision_recall_curve
    sklearn_stub.isotonic = isotonic_module
    sklearn_stub.metrics = metrics_module
    sklearn_stub.model_selection = model_selection_module
    sys.modules["sklearn"] = sklearn_stub
    sys.modules["sklearn.isotonic"] = isotonic_module
    sys.modules["sklearn.metrics"] = metrics_module
    sys.modules["sklearn.model_selection"] = model_selection_module
