#!/usr/bin/env python3
"""Shared utilities for visualization."""

import os

import pandas as pd
import seaborn as sns

RESULTS_DIR = os.path.join('experiments', 'results')

sns.set_theme(
    style='whitegrid',
    palette='husl',
    font_scale=1.1,
)


def build_viz_df(results: dict) -> pd.DataFrame:
    """
    Build DataFrame for visualizations.

    Args:
        results: Dict of correction DataFrames.

    Returns:
        Summary DataFrame for plotting.
    """
    rows = []
    for key, df in results.items():
        if len(df) == 0:
            continue
        llm, asr = key.split('__')
        bl = df['wer_baseline'].mean()
        cr = df['wer_corrected'].mean()
        rows.append({
            'LLM': llm,
            'ASR': asr,
            'WER Change (%)': (bl - cr) / bl * 100,
            'Baseline WER': bl,
            'Corrected WER': cr,
            'Improved (%)': (
                df['wer_improved'].mean() * 100
            ),
            'Degraded (%)': (
                df['wer_degraded'].mean() * 100
            ),
        })

    return pd.DataFrame(rows)
