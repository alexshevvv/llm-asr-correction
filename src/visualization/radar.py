#!/usr/bin/env python3
"""LLM performance radar chart."""

import logging
import os
from math import pi

import matplotlib.pyplot as plt
import seaborn as sns

from src.visualization.common import RESULTS_DIR
from src.visualization.common import build_viz_df

logger = logging.getLogger(__name__)


def plot_radar(results: dict) -> None:
    """
    Plot LLM performance radar chart.

    Args:
        results: Dict of correction DataFrames.
    """
    viz_df = build_viz_df(results)
    pivot = viz_df.pivot(
        index='LLM',
        columns='ASR',
        values='WER Change (%)',
    )

    categories = list(pivot.columns)
    n_cats = len(categories)
    angles = [
        n / float(n_cats) * 2 * pi
        for n in range(n_cats)
    ]
    angles += angles[:1]

    fig, ax = plt.subplots(
        figsize=(10, 10),
        subplot_kw=dict(polar=True),
    )
    colors = sns.color_palette('husl', len(pivot))

    for i, (llm, row) in enumerate(pivot.iterrows()):
        values = row.values.tolist()
        values += values[:1]
        ax.plot(
            angles,
            values,
            'o-',
            linewidth=2,
            label=llm,
            color=colors[i],
        )
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_title(
        'LLM Performance Radar\n'
        '(WER Change % per ASR)',
        fontsize=14,
        fontweight='bold',
        y=1.08,
    )

    ax.legend(
        loc='upper right',
        bbox_to_anchor=(1.3, 1.1),
        fontsize=10,
    )
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, 'plot_radar.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    logger.info('Saved: %s', path)
    plt.close(fig)
