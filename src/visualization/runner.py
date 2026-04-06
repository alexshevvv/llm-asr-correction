#!/usr/bin/env python3
"""Run all visualization plots."""

import logging
import os

from src.visualization.baseline import plot_baseline_comparison
from src.visualization.common import RESULTS_DIR
from src.visualization.corrected_wer import plot_corrected_wer
from src.visualization.diverging import plot_diverging
from src.visualization.heatmap import plot_heatmap
from src.visualization.radar import plot_radar
from src.visualization.scatter import plot_scatter
from src.visualization.wer_change import plot_wer_change

logger = logging.getLogger(__name__)


def plot_all(
    results: dict,
    baselines: dict,
) -> None:
    """
    Generate all plots.

    Args:
        results: Dict of correction DataFrames.
        baselines: Dict of baseline DataFrames.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_baseline_comparison(baselines)
    plot_wer_change(results)
    plot_corrected_wer(results)
    plot_diverging(results)
    plot_heatmap(results)
    plot_scatter(results)
    plot_radar(results)
    logger.info('All 7 plots saved to %s', RESULTS_DIR)
