#!/usr/bin/env python3
"""Diverging horizontal bar chart."""

import logging
import os

import matplotlib.pyplot as plt

from src.visualization.common import RESULTS_DIR
from src.visualization.common import build_viz_df

logger = logging.getLogger(__name__)


def plot_diverging(results: dict) -> None:
    """
    Plot all experiments sorted by WER change.

    Args:
        results: Dict of correction DataFrames.
    """
    viz_df = build_viz_df(results)
    viz_df['Label'] = (
        viz_df['LLM'] + ' -> ' + viz_df['ASR']
    )
    viz_sorted = viz_df.sort_values('WER Change (%)')

    colors = [
        '#2ecc71' if v > 0 else '#e74c3c'
        for v in viz_sorted['WER Change (%)']
    ]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.barh(
        viz_sorted['Label'],
        viz_sorted['WER Change (%)'],
        color=colors,
        edgecolor='white',
        height=0.7,
    )
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_title(
        'All Experiments: WER Change\n'
        '(green = improved, red = degraded)',
        fontsize=14,
        fontweight='bold',
    )
    for i, (val, label) in enumerate(
        zip(
            viz_sorted['WER Change (%)'],
            viz_sorted['Label'],
        )
    ):
        ax.text(
            val + (0.5 if val >= 0 else -0.5), i,
            f'{val:+.1f}%',
            ha='left' if val >= 0 else 'right',
            va='center',
            fontsize=8,
            fontweight='bold',
        )
    plt.tight_layout()
    path = os.path.join(
        RESULTS_DIR, 'plot_diverging.png',
    )
    fig.savefig(path, dpi=150)
    logger.info('Saved: %s', path)
    plt.close(fig)
