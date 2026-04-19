#!/usr/bin/env python3
"""LLM performance radar chart."""

import logging
import os
from math import pi

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.visualization.common import RESULTS_DIR

logger = logging.getLogger(__name__)

RADAR_MIN = -200
RADAR_MAX = 100


def plot_radar(analysis_df: pd.DataFrame) -> None:
    """
    Plot LLM performance radar aggregated by Dataset.

    Args:
        analysis_df: Unified analysis DataFrame.
    """
    radar_data = (
        analysis_df
        .groupby(['LLM', 'Dataset'])['WER Change (%)']
        .mean()
        .reset_index()
    )
    radar_data['clipped'] = radar_data['WER Change (%)'].clip(
        lower=RADAR_MIN, upper=RADAR_MAX,
    )
    pivot = radar_data.pivot(
        index='LLM', columns='Dataset', values='clipped',
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
        values = row.values.tolist() + row.values[:1].tolist()
        ax.plot(
            angles, values, 'o-',
            linewidth=2, label=llm, color=colors[i],
        )
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.plot(
        angles, [0] * len(angles),
        color='black', linewidth=1,
        linestyle='--', alpha=0.6,
    )
    ax.set_ylim(RADAR_MIN, RADAR_MAX)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title(
        'LLM Performance Radar by Dataset\n'
        f'(clipped to [{RADAR_MIN}%, {RADAR_MAX}%])',
        fontsize=14, fontweight='bold', y=1.08,
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
