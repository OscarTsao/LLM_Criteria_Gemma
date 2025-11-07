"""Evaluate a trained fold checkpoint with optional calibration and aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.data import LABEL_NAMES
from src.eval.aggregate import aggregate_labels, aggregate_probabilities
from src.eval.calibration import calibrate
from src.eval.report import generate_report
from src.eval.thresholds import search_thresholds

DEFAULT_CFG = OmegaConf.create(
    {
        "ckpt_dir": "",
        "output_dir": None,
        "agg": {"strategy": "max", "temperature": 1.0},
        "calib": {
            "method": "none",
            "temperature_init": 1.0,
            "isotonic_out_of_bounds": "clip",
        },
        "thresholds": {"grid_size": 1001},
        "ece_bins": 15,
    }
)


def load_predictions(pred_dir: Path) -> Dict[str, Any]:
    logits_path = pred_dir / "oof_logits.npy"
    probs_path = pred_dir / "oof_probs.npy"
    labels_path = pred_dir / "oof_labels.npy"
    meta_path = pred_dir / "ids.json"

    logits = np.load(logits_path) if logits_path.exists() else None
    probs = np.load(probs_path) if probs_path.exists() else None
    if probs is None and logits is not None:
        probs = 1.0 / (1.0 + np.exp(-logits))
    if probs is None:
        raise FileNotFoundError(f"Could not locate probabilities in {pred_dir}.")
    labels = np.load(labels_path)
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    return {"logits": logits, "probabilities": probs, "labels": labels, "meta": meta}


def main() -> None:
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(DEFAULT_CFG, cli_cfg)
    if not cfg.ckpt_dir:
        raise ValueError("ckpt_dir must be provided.")
    ckpt_dir = Path(cfg.ckpt_dir).expanduser().resolve()
    pred_dir = ckpt_dir / "predictions"
    if not pred_dir.exists():
        pred_dir = ckpt_dir

    data = load_predictions(pred_dir)
    logits = data["logits"]
    probabilities = data["probabilities"]
    labels = data["labels"]
    meta = data["meta"]

    calibration = calibrate(
        logits=logits,
        probabilities=probabilities,
        labels=labels,
        method=cfg.calib.method,
        temperature_init=cfg.calib.temperature_init,
        isotonic_out_of_bounds=cfg.calib.isotonic_out_of_bounds,
    )
    calibrated_probs = calibration.probabilities

    agg_result = aggregate_probabilities(
        calibrated_probs,
        meta=meta,
        strategy=cfg.agg.strategy,
        temperature=cfg.agg.temperature,
    )
    post_labels = aggregate_labels(labels, meta)

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

    if cfg.output_dir:
        out_dir = Path(cfg.output_dir).expanduser().resolve()
    else:
        out_dir = ckpt_dir / "eval" / f"{cfg.agg.strategy}_{cfg.calib.method}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "aggregation": {
            "strategy": cfg.agg.strategy,
            "temperature": cfg.agg.temperature,
        },
        "calibration": {
            "method": calibration.method,
            "parameters": calibration.parameters,
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
        json.dump(metrics_payload, handle, indent=2)

    pd.DataFrame(report["per_class"]).to_csv(out_dir / "per_class.csv", index=False)
    pd.DataFrame(report["coverage_risk"]).to_csv(out_dir / "coverage_risk.csv", index=False)
    pd.DataFrame(report["reliability"]).to_csv(out_dir / "reliability_curve.csv", index=False)

    print(
        f"Aggregated macro-AUPRC={report['macro_auprc']:.4f} "
        f"macro-F1(global)={report['macro_f1_global']:.4f} "
        f"macro-F1(per-class)={report['macro_f1_per_class']:.4f}"
    )


if __name__ == "__main__":
    main()
