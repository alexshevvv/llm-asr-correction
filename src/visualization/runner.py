#!/usr/bin/env python3
"""Run all visualization plots."""

import logging
import os

import pandas as pd

from src.visualization.baseline import plot_baseline_comparison
from src.visualization.common import RESULTS_DIR
from src.visualization.heatmap import plot_heatmap
from src.visualization.radar import plot_radar
from src.visualization.scatter import plot_scatter
from src.visualization.stacked_bar import plot_stacked_bar
from src.visualization.wer_change import plot_wer_change

logger = logging.getLogger(__name__)


def plot_all(
    analysis_df: pd.DataFrame,
    baselines: dict[str, pd.DataFrame],
) -> None:
    """
    Generate all plots.

    Args:
        analysis_df: Unified analysis DataFrame.
        baselines: Dict of baseline DataFrames.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_baseline_comparison(baselines)
    plot_wer_change(analysis_df)
    plot_heatmap(analysis_df)
    plot_scatter(analysis_df)
    plot_radar(analysis_df)
    plot_stacked_bar(analysis_df)
    logger.info('All plots saved to %s', RESULTS_DIR)
