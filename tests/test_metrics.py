import numpy as np

from src.eval.thresholds import search_thresholds
from src.training.metrics import (
    compute_macro_auprc,
    compute_macro_f1,
    compute_per_class_pr,
)


def test_macro_metrics_and_thresholds():
    y_true = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.int32,
    )
    y_score = np.array(
        [
            [0.9, 0.1, 0.2],
            [0.2, 0.8, 0.1],
            [0.7, 0.6, 0.2],
            [0.1, 0.2, 0.9],
        ],
        dtype=np.float32,
    )
    macro_auprc = compute_macro_auprc(y_true, y_score)
    assert 0.0 <= macro_auprc <= 1.0

    macro_f1_result = compute_macro_f1(y_true, y_score, threshold=0.5, label_names=("A", "B", "C"))
    assert "macro" in macro_f1_result and macro_f1_result["macro"] > 0.0
    assert set(macro_f1_result["per_class"].keys()) == {"A", "B", "C"}

    pr_curves = compute_per_class_pr(y_true, y_score, label_names=("A", "B", "C"), max_points=10)
    assert pr_curves["A"]["precision"][0] <= 1.0

    thresholds = search_thresholds(y_score, y_true, label_names=("A", "B", "C"), grid_size=21)
    assert 0.0 <= thresholds.global_threshold <= 1.0
    for label, thr in thresholds.per_class_thresholds.items():
        assert 0.0 <= thr <= 1.0
        assert thresholds.per_class_f1[label] >= 0.0
