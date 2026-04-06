#!/usr/bin/env python3
"""Scatter: Baseline WER vs LLM effectiveness."""

import logging
import os

import matplotlib.pyplot as plt
import seaborn as sns

from src.visualization.common import RESULTS_DIR
from src.visualization.common import build_viz_df

logger = logging.getLogger(__name__)


def plot_scatter(results: dict) -> None:
    """
    Plot baseline WER vs correction effectiveness.

    Args:
        results: Dict of correction DataFrames.
    """
    viz_df = build_viz_df(results)
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.scatterplot(
        data=viz_df,
        x='Baseline WER',
        y='WER Change (%)',
        hue='LLM',
        style='LLM',
        s=200,
        ax=ax,
        edgecolor='black',
        linewidth=0.5,
    )

    for _, row in viz_df.iterrows():
        ax.annotate(
            row['ASR'],
            (row['Baseline WER'], row['WER Change (%)']),
            fontsize=7,
            alpha=0.7,
            xytext=(5, 5),
            textcoords='offset points',
        )

    ax.axhline(y=0, color='black', ls='--', lw=1)
    ax.axvspan(
        0, 0.10,
        alpha=0.1,
        color='red',
        label='Danger zone (WER < 10%)',
    )
    ax.axvspan(
        0.10, ax.get_xlim()[1],
        alpha=0.1,
        color='green',
        label='Effective zone (WER > 10%)',
    )
    ax.set_title(
        'Baseline WER vs LLM Correction Effectiveness',
        fontsize=14,
        fontweight='bold',
    )

    ax.set_xlabel('Baseline WER', fontsize=12)
    ax.set_ylabel('WER Change (%)', fontsize=12)
    ax.legend(fontsize=9, loc='best')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'plot_scatter.png')
    fig.savefig(path, dpi=150)
    logger.info('Saved: %s', path)
    plt.close(fig)
