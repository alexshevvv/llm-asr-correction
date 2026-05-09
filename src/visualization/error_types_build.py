#!/usr/bin/env python3
"""Build per-sample error classification DataFrame."""

import pandas as pd

from src.evaluation.error_classification import (
    classify_errors,
)


def build_error_df(
    baselines: dict[str, pd.DataFrame],
    results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build DataFrame with S/I/D counts per sample.

    """
    rows = []
    for key, df in baselines.items():
        parts = key.split('__')
        exp = f'{parts[0]} / {parts[1]}' if len(parts) > 1 else key
        for _, row in df.iterrows():
            errors = classify_errors(
                row['reference'], row['hypothesis'],
            )
            rows.append({
                'Stage': 'Baseline',
                'LLM': 'none',
                'Experiment': exp,
                **errors,
            })

    for key, df in results.items():
        if len(df) == 0:
            continue
        parts = key.split('__')
        if len(parts) != 3:
            continue
        for _, row in df.iterrows():
            errors = classify_errors(
                row['reference'], row['corrected'],
            )
            rows.append({
                'Stage': 'Corrected',
                'LLM': parts[0],
                'Experiment': f'{parts[1]} / {parts[2]}',
                **errors,
            })
    return pd.DataFrame(rows)
