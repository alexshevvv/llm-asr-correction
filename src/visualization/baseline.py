#!/usr/bin/env python3
"""ASR baseline comparison plots by ASR family."""

import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.visualization.common import RESULTS_DIR

logger = logging.getLogger(__name__)


def plot_baseline_comparison(
    baselines: dict[str, pd.DataFrame],
) -> None:
    """
    Plot ASR baseline WER and error distribution.

    Args:
        baselines: Dict '<asr>__<dataset>' -> DataFrame
            with 'wer' column.
    """
    rows = []
    for key, df in baselines.items():
        parts = key.split('__')
        asr_key = parts[0] if parts else key
        dataset_key = parts[1] if len(parts) > 1 else ''
        rows.append({
            'Experiment': f'{asr_key} / {dataset_key}',
            'ASR': asr_key,
            'Dataset': dataset_key,
            'Mean WER': df['wer'].mean(),
            'Errors': int((df['wer'] > 0).sum()),
            'Perfect': int((df['wer'] == 0).sum()),
        })
    view = pd.DataFrame(rows)
    fig, axes = plt.subplots(
        1, 2, figsize=(max(10, len(view) * 2), 5),
    )

    sns.barplot(
        data=view,
        x='Experiment',
        y='Mean WER',
        hue='Experiment',
        ax=axes[0],
        palette='viridis',
        edgecolor='white',
        legend=False,
        errorbar=None,
    )
    axes[0].set_title(
        'ASR Baseline: Mean WER',
        fontsize=13, fontweight='bold',
    )
    for container in axes[0].containers:
        axes[0].bar_label(
            container, fmt='%.3f', fontsize=9,
        )
    axes[0].tick_params(axis='x', rotation=45)
    for label in axes[0].get_xticklabels():
        label.set_ha('right')
    axes[0].set_xlabel('')

    bl_melt = view.melt(
        id_vars='Experiment',
        value_vars=['Perfect', 'Errors'],
        var_name='Type',
        value_name='Count',
    )
    sns.barplot(
        data=bl_melt,
        x='Experiment',
        y='Count',
        hue='Type',
        ax=axes[1],
        palette=['#2ecc71', '#e74c3c'],
        edgecolor='white',
        errorbar=None,
    )
    axes[1].set_title(
        'Perfect vs Error Samples',
        fontsize=13, fontweight='bold',
    )
    axes[1].tick_params(axis='x', rotation=45)
    for label in axes[1].get_xticklabels():
        label.set_ha('right')
    axes[1].set_xlabel('')
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(
        RESULTS_DIR, 'plot_baselines.png',
    )
    fig.savefig(path, dpi=150)
    logger.info('Saved: %s', path)
    plt.close(fig)
