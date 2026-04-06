#!/usr/bin/env python3
"""ASR baseline comparison plot."""

import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.visualization.common import RESULTS_DIR

logger = logging.getLogger(__name__)


def plot_baseline_comparison(
    baselines: dict,
) -> None:
    """
    Plot ASR baseline WER + error distribution.

    Args:
        baselines: Dict name -> DataFrame.
    """
    data = []
    for name, df in baselines.items():
        data.append({
            'ASR Model': name,
            'Mean WER': df['wer'].mean(),
            'Samples with Errors': int((df['wer'] > 0).sum()),
            'Perfect (WER=0)': int((df['wer'] == 0).sum()),
        })
    bl_df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    sns.barplot(
        data=bl_df,
        x='ASR Model',
        y='Mean WER',
        hue='ASR Model',
        ax=axes[0],
        palette='viridis',
        edgecolor='white',
        legend=False,
    )
    axes[0].set_title(
        'ASR Baseline: Mean WER',
        fontsize=13,
        fontweight='bold',
    )
    axes[0].bar_label(
        axes[0].containers[0],
        fmt='%.2f%%',
        fontsize=9,
    )
    axes[0].tick_params(axis='x', rotation=25)

    bl_melt = bl_df.melt(
        id_vars='ASR Model',
        value_vars=[
            'Perfect (WER=0)',
            'Samples with Errors',
        ],
        var_name='Type',
        value_name='Count',
    )
    sns.barplot(
        data=bl_melt,
        x='ASR Model',
        y='Count',
        hue='Type',
        ax=axes[1],
        palette=['#2ecc71', '#e74c3c'],
        edgecolor='white',
    )
    axes[1].set_title(
        'ASR: Perfect vs Error Samples',
        fontsize=13,
        fontweight='bold',
    )
    axes[1].tick_params(axis='x', rotation=25)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(
        RESULTS_DIR, 'plot_baselines.png',
    )
    fig.savefig(path, dpi=150)
    logger.info('Saved: %s', path)
    plt.close(fig)

