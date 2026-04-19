#!/usr/bin/env python3
"""Stacked bar: Improved / Degraded / Unchanged by ASR family."""

import logging
import os

import pandas as pd

from src.visualization.common import ASR_GROUPS
from src.visualization.common import RESULTS_DIR
from src.visualization.stacked_bar_draw import draw_stacked

logger = logging.getLogger(__name__)

COLORS = {
    'Improved': '#2ecc71',
    'Degraded': '#e74c3c',
    'Unchanged': '#95a5a6',
}


def plot_stacked_bar(
    analysis_df: pd.DataFrame,
) -> None:
    """
    Plot sample-level outcomes per ASR family.

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
            group_df = group_df.assign(
                X=group_df['Dataset'],
            )
        else:
            group_df = group_df.assign(
                X=(
                    group_df['ASR'] + ' / '
                    + group_df['Dataset']
                ),
            )

        safe = group_name.lower().replace(' ', '_')
        path = os.path.join(
            RESULTS_DIR, f'plot_stacked_{safe}.png',
        )
        draw_stacked(group_df, group_name, path)
        logger.info('Saved: %s', path)
