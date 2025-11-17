"""Utilities for working with ReDSM5 annotation files."""

from __future__ import annotations

from typing import Tuple

import pandas as pd


_SYMPTOM_COLUMN_PREFERENCES: Tuple[str, ...] = ('symptom_label', 'DSM5_symptom')


def ensure_symptom_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize symptom label column to the canonical 'symptom_label'.

    The dataset CSVs have shipped with both 'symptom_label' and the legacy
    'DSM5_symptom' column names. This helper lets callers work against the
    canonical name regardless of which version is on disk.
    """
    for column_name in _SYMPTOM_COLUMN_PREFERENCES:
        if column_name in df.columns:
            if column_name != 'symptom_label':
                df = df.rename(columns={column_name: 'symptom_label'})
            return df

    available_columns = ', '.join(df.columns)
    raise KeyError(
        "redsm5_annotations.csv must expose a symptom label column "
        f"(available columns: {available_columns})"
    )
