#!/usr/bin/env python3
"""Baseline vs Corrected WER faceted plot."""

import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.visualization.common import RESULTS_DIR

logger = logging.getLogger(__name__)


def plot_corrected_wer(results: dict) -> None:
    """
    Plot Baseline vs Corrected WER by LLM.

    Args:
        results: Dict of correction DataFrames.
    """
    melt_rows = []
    for key, df in results.items():
        if len(df) == 0:
            continue
        llm, asr = key.split('__')
        melt_rows.append({
            'LLM': llm, 'ASR': asr,
            'Stage': 'Baseline',
            'WER': df['wer_baseline'].mean(),
        })
        melt_rows.append({
            'LLM': llm, 'ASR': asr,
            'Stage': 'Corrected',
            'WER': df['wer_corrected'].mean(),
        })
    melt_df = pd.DataFrame(melt_rows)

    graph = sns.catplot(
        data=melt_df,
        x='ASR',
        y='WER',
        hue='Stage',
        col='LLM',
        kind='bar',
        col_wrap=2,
        height=5,
        aspect=1.4,
        palette=['#e74c3c', '#2ecc71'],
        edgecolor='white',
    )

    graph.set_titles(
        '{col_name}', fontsize=13, fontweight='bold',
    )
    graph.set_axis_labels('ASR', 'Mean WER', fontsize=11)

    for ax in graph.axes.flat:
        ax.tick_params(axis='x', rotation=20)

    graph.fig.suptitle(
        'Baseline vs Corrected WER by LLM',
        fontsize=15, fontweight='bold', y=1.02,
    )

    plt.tight_layout()
    path = os.path.join(
        RESULTS_DIR, 'plot_corrected_wer.png',
    )

    graph.fig.savefig(path, dpi=150, bbox_inches='tight')
    logger.info('Saved: %s', path)
    plt.close(graph.fig)
