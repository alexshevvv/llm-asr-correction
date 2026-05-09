#!/usr/bin/env python3
"""Baseline error types bar chart."""

import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

ETYPE_COLORS = ['#e74c3c', '#3498db', '#f39c12']


def draw_baseline_errors(
    error_df: pd.DataFrame,
    results_dir: str,
) -> None:
    """Plot baseline error types by experiment."""
    bl = error_df[error_df['Stage'] == 'Baseline']
    agg = (
        bl.groupby('Experiment')[
            ['substitutions', 'insertions', 'deletions']
        ]
        .sum()
        .reset_index()
    )
    melt = agg.melt(
        id_vars='Experiment',
        var_name='Error Type',
        value_name='Count',
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=melt, x='Experiment', y='Count',
        hue='Error Type', ax=ax,
        palette=ETYPE_COLORS, edgecolor='white',
    )
    ax.set_title(
        'ASR Baseline: Error Types',
        fontsize=13, fontweight='bold',
    )
    ax.set_xlabel('')
    ax.legend(title='Error Type', fontsize=9)
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    path = os.path.join(
        results_dir, 'plot_error_types_baseline.png',
    )
    fig.savefig(path, dpi=150)
    logger.info('Saved: %s', path)
    plt.close(fig)
