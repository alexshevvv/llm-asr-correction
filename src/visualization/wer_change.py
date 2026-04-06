#!/usr/bin/env python3
"""WER change grouped barplot."""

import logging
import os

import matplotlib.pyplot as plt
import seaborn as sns

from src.visualization.common import RESULTS_DIR
from src.visualization.common import build_viz_df

logger = logging.getLogger(__name__)


def plot_wer_change(results: dict) -> None:
    """
    Plot WER change grouped barplot.

    Args:
        results: Dict of correction DataFrames.
    """
    viz_df = build_viz_df(results)
    fig, ax = plt.subplots(figsize=(18, 7))
    sns.barplot(
        data=viz_df,
        x='ASR',
        y='WER Change (%)',
        hue='LLM',
        ax=ax,
        edgecolor='white',
    )
    ax.axhline(y=0, color='black', linewidth=1, ls='--')
    ax.set_title(
        'LLM Benchmark: WER Change by Model and ASR\n'
        '(positive = improvement, '
        'negative = degradation)',
        fontsize=14,
        fontweight='bold',
    )
    ax.set_xlabel('ASR Experiment', fontsize=12)
    ax.set_ylabel('WER Change (%)', fontsize=12)
    ax.legend(
        title='LLM Model',
        fontsize=10,
        title_fontsize=11,
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
