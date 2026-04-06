#!/usr/bin/env python3
"""WER improvement heatmap."""

import logging
import os

import matplotlib.pyplot as plt
import seaborn as sns

from src.visualization.common import RESULTS_DIR
from src.visualization.common import build_viz_df

logger = logging.getLogger(__name__)


def plot_heatmap(results: dict) -> None:
    """
    Plot WER improvement heatmap.

    Args:
        results: Dict of correction DataFrames.
    """
    viz_df = build_viz_df(results)
    pivot = viz_df.pivot(
        index='LLM',
        columns='ASR',
        values='WER Change (%)',
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt='+.1f',
        cmap='RdYlGn',
        center=0,
        linewidths=2,
        linecolor='white',
        ax=ax,
        annot_kws={
            'fontsize': 12,
            'fontweight': 'bold',
        },
        cbar_kws={'label': 'WER Change (%)'},
    )
    ax.set_title(
        'WER Improvement Heatmap\n'
        '(green = improved, red = degraded)',
        fontsize=14,
        fontweight='bold',
    )
    ax.set_xlabel('ASR Experiment', fontsize=12)
    ax.set_ylabel('LLM Model', fontsize=12)
    plt.xticks(rotation=20)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'plot_heatmap.png')
    fig.savefig(path, dpi=150)
    logger.info('Saved: %s', path)
    plt.close(fig)
