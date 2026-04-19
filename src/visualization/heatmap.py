#!/usr/bin/env python3
"""WER improvement heatmaps split by ASR family."""

import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.visualization.common import ASR_GROUPS
from src.visualization.common import RESULTS_DIR

logger = logging.getLogger(__name__)


def plot_heatmap(analysis_df: pd.DataFrame) -> None:
    """
    Plot WER improvement heatmaps per ASR family.

    Args:
        analysis_df: Unified analysis DataFrame.
    """
    for group_name, asr_list in ASR_GROUPS.items():
        group_df = analysis_df[
            analysis_df['ASR'].isin(asr_list)
        ]
        if len(group_df) == 0:
            continue

        if len(asr_list) == 1:
            pivot = group_df.pivot(
                index='LLM',
                columns='Dataset',
                values='WER Change (%)',
            )
        else:
            pivot = group_df.pivot(
                index='LLM',
                columns='Experiment',
                values='WER Change (%)',
            )

        n_cols = len(pivot.columns)
        fig, ax = plt.subplots(
            figsize=(max(8, n_cols * 2), 5),
        )
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
                'fontsize': 11, 'fontweight': 'bold',
            },
            cbar_kws={'label': 'WER Change (%)'},
        )
        ax.set_title(
            f'WER Heatmap: {group_name}\n'
            '(green = improved, red = degraded)',
            fontsize=13,
            fontweight='bold',
        )
        ax.set_ylabel('LLM Model', fontsize=11)
        plt.xticks(rotation=25, ha='right')
        plt.tight_layout()

        safe_name = group_name.lower().replace(' ', '_')
        path = os.path.join(
            RESULTS_DIR,
            f'plot_heatmap_{safe_name}.png',
        )
        fig.savefig(path, dpi=150)
        logger.info('Saved: %s', path)
        plt.close(fig)

