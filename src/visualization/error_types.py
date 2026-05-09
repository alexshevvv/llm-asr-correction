#!/usr/bin/env python3
"""Error type analysis."""

import logging

import pandas as pd

from src.visualization.common import RESULTS_DIR
from src.visualization.error_types_build import (
    build_error_df,
)
from src.visualization.error_types_draw_baseline import (
    draw_baseline_errors,
)
from src.visualization.error_types_draw_delta import (
    draw_error_delta,
)

logger = logging.getLogger(__name__)


def plot_error_types(
    baselines: dict[str, pd.DataFrame],
    results: dict[str, pd.DataFrame],
) -> None:
    """
    Build error classification and generate plots.

    """
    error_df = build_error_df(baselines, results)
    draw_baseline_errors(error_df, RESULTS_DIR)
    draw_error_delta(results, RESULTS_DIR)

    summary = (
        error_df
        .groupby(['Stage', 'LLM'])[
            ['substitutions', 'insertions', 'deletions']
        ]
        .mean()
        .round(2)
        .reset_index()
    )
    logger.info(
        'Error Type Summary:\n%s',
        summary.to_string(index=False),
    )