#!/usr/bin/env python3
"""Save benchmark results to disk."""

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def save_correction_csvs(
    results: dict[str, pd.DataFrame],
    results_dir: str,
) -> None:
    """
    Save each correction DataFrame as a separate CSV.

    """
    os.makedirs(results_dir, exist_ok=True)
    for key, df in results.items():
        safe_name = key.replace(' ', '_')
        path = os.path.join(
            results_dir, f'correction_{safe_name}.csv',
        )
        df.to_csv(path, index=False)
    logger.info(
        'Saved %d correction CSVs to %s',
        len(results), results_dir,
    )
