#!/usr/bin/env python3
"""Save benchmark results to disk."""

import json
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


def save_run_config(
    configs_dir: str,
    timestamp: str,
    max_samples: int,
    baselines_info: dict,
) -> str:
    """
    Save experiment config as JSON.

    """
    os.makedirs(configs_dir, exist_ok=True)
    run_config = {
        'timestamp': timestamp,
        'max_samples': max_samples,
        'baselines': baselines_info,
    }
    filename = f'run_{timestamp}.json'
    path = os.path.join(configs_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(
            run_config, f,
            indent=2, ensure_ascii=False,
        )
    logger.info('Config saved: %s', path)
    return path
