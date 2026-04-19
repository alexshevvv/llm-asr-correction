#!/usr/bin/env python3
"""WER change grouped barplot by Dataset."""

import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.visualization.common import RESULTS_DIR

logger = logging.getLogger(__name__)


def plot_wer_change(analysis_df: pd.DataFrame) -> None:
    """
    Plot mean WER change by LLM x Dataset.

    Aggregates WER Change across all ASR models for
    each (LLM, Dataset) pair.

    Args:
        analysis_df: Unified analysis DataFrame.
    """
    agg = (
        analysis_df
        .groupby(['LLM', 'Dataset'])['WER Change (%)']
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(18, 7))
    sns.barplot(
        data=agg,
        x='Dataset',
        y='WER Change (%)',
        hue='LLM',
        ax=ax,
        edgecolor='white',
    )
    ax.axhline(y=0, color='black', linewidth=1, ls='--')
    ax.set_title(
        'LLM Benchmark: Mean WER Change by Dataset\n'
        '(averaged across ASR; '
        'positive = improvement)',
        fontsize=14,
        fontweight='bold',
    )
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Mean WER Change (%)', fontsize=12)
    ax.legend(
        title='LLM Model',
        fontsize=9,
        title_fontsize=10,
        loc='best',
        ncol=2,
    )
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt='%.1f%%',
            fontsize=7,
            padding=3,
        )
    plt.xticks(rotation=15)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, 'plot_wer_change.png')

    fig.savefig(path, dpi=150)
    logger.info('Saved: %s', path)
    plt.close(fig)
