#!/usr/bin/env python3
"""Scatter: Baseline WER vs LLM effectiveness."""

import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.visualization.common import RESULTS_DIR

logger = logging.getLogger(__name__)

OUTLIER_MIN = -200


def plot_scatter(analysis_df: pd.DataFrame) -> None:
    """
    Plot baseline WER vs correction effectiveness.

    Args:
        analysis_df: Unified analysis DataFrame.
    """
    n_outliers = int(
        (analysis_df['WER Change (%)'] < OUTLIER_MIN).sum()
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=analysis_df,
        x='Baseline WER',
        y='WER Change (%)',
        hue='LLM',
        style='LLM',
        s=150,
        ax=ax,
        edgecolor='black',
        linewidth=0.5,
        alpha=0.75,
    )
    ax.axhline(y=0, color='black', ls='--', lw=1)
    ax.axvspan(
        0, 0.10, alpha=0.1, color='red',
        label='Danger zone (WER < 10%)',
    )
    ax.axvspan(
        0.10, ax.get_xlim()[1], alpha=0.1, color='green',
        label='Effective zone (WER > 10%)',
    )
    ax.set_ylim(OUTLIER_MIN, 100)

    title = 'Baseline WER vs LLM Correction Effectiveness'
    if n_outliers > 0:
        title += (
            f'\n({n_outliers} outliers below '
            f'{OUTLIER_MIN}% clipped)'
        )
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Baseline WER', fontsize=12)
    ax.set_ylabel('WER Change (%)', fontsize=12)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, 'plot_scatter.png')
    fig.savefig(path, dpi=150)
    logger.info('Saved: %s', path)
    plt.close(fig)
