"""Merge OOF predictions across folds and runs to compute global metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.data import LABEL_NAMES
from src.eval.aggregate import aggregate_labels, aggregate_probabilities
from src.eval.report import generate_report
from src.eval.thresholds import search_thresholds

DEFAULT_CFG = OmegaConf.create(
    {
        "runs": [],
        "output_dir": "outputs/summary",
        "agg": {"strategy": "max", "temperature": 1.0},
        "thresholds": {"grid_size": 1001},
        "ece_bins": 15,
    }
)


def discover_fold_paths(run_dir: Path) -> List[Path]:
    folds = sorted(p for p in run_dir.glob("fold_*") if p.is_dir())
    if not folds:
        raise FileNotFoundError(f"No fold_* directories found under {run_dir}.")
    return folds


def load_fold_predictions(fold_dir: Path) -> Dict[str, Any]:
    pred_dir = fold_dir / "predictions"
    logits_path = pred_dir / "oof_logits.npy"
    probs_path = pred_dir / "oof_probs.npy"
    labels_path = pred_dir / "oof_labels.npy"
    meta_path = pred_dir / "ids.json"

    logits = np.load(logits_path) if logits_path.exists() else None
    probs = np.load(probs_path) if probs_path.exists() else None
    if probs is None and logits is not None:
        probs = 1.0 / (1.0 + np.exp(-logits))
    if probs is None:
        raise FileNotFoundError(f"Missing probabilities in {pred_dir}.")
    labels = np.load(labels_path)
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    return {"logits": logits, "probabilities": probs, "labels": labels, "meta": meta}


def main() -> None:
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(DEFAULT_CFG, cli_cfg)
    run_dirs = [Path(run).expanduser().resolve() for run in cfg.runs]
    if not run_dirs:
        raise ValueError("At least one run directory must be provided via runs=<path>.")

    logits_list: List[np.ndarray] = []
    probs_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    meta_list: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        for fold_dir in discover_fold_paths(run_dir):
            data = load_fold_predictions(fold_dir)
            if data["logits"] is not None:
                logits_list.append(data["logits"])
            probs_list.append(data["probabilities"])
            labels_list.append(data["labels"])
            meta_list.extend(data["meta"])

    logits = np.vstack(logits_list) if logits_list else None
    probabilities = np.vstack(probs_list)
    labels = np.vstack(labels_list)

    agg_result = aggregate_probabilities(
        probabilities,
        meta_list,
        strategy=cfg.agg.strategy,
        temperature=cfg.agg.temperature,
    )
    post_labels = aggregate_labels(labels, meta_list)

    threshold_result = search_thresholds(
        agg_result.probabilities,
        post_labels,
        label_names=LABEL_NAMES,
        grid_size=cfg.thresholds.grid_size,
    )
    report = generate_report(
        probabilities=agg_result.probabilities,
        labels=post_labels,
        label_names=LABEL_NAMES,
        global_threshold=threshold_result.global_threshold,
        per_class_thresholds=threshold_result.per_class_thresholds,
        per_class_f1=threshold_result.per_class_f1,
        macro_f1_per_class=threshold_result.macro_f1_per_class,
        ece_bins=cfg.ece_bins,
    )

    out_dir = Path(cfg.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "oof_probs.npy", probabilities)
    np.save(out_dir / "oof_labels.npy", labels)
    with (out_dir / "ids.json").open("w", encoding="utf-8") as handle:
        json.dump(meta_list, handle, indent=2)

    payload = {
        "aggregation": {
            "strategy": cfg.agg.strategy,
            "temperature": cfg.agg.temperature,
        },
        "thresholds": {
            "global": threshold_result.global_threshold,
            "macro_f1_global": threshold_result.macro_f1_global,
            "per_class": threshold_result.per_class_thresholds,
            "macro_f1_per_class": threshold_result.macro_f1_per_class,
            "per_class_f1": threshold_result.per_class_f1,
        },
        "metrics": report,
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    pd.DataFrame(report["per_class"]).to_csv(out_dir / "per_class.csv", index=False)
    pd.DataFrame(report["coverage_risk"]).to_csv(out_dir / "coverage_risk.csv", index=False)

    print(
        f"Merged macro-AUPRC={report['macro_auprc']:.4f} "
        f"macro-F1(global)={report['macro_f1_global']:.4f} "
        f"macro-F1(per-class)={report['macro_f1_per_class']:.4f}"
    )


if __name__ == "__main__":
    main()
