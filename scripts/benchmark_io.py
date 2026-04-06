#!/usr/bin/env python3
"""Benchmark I/O: save results, configs, dirs."""

import json
import logging
import os
from datetime import datetime

import pandas as pd

from scripts.benchmark_config import CONFIGS_DIR
from scripts.benchmark_config import DATA_DIR
from scripts.benchmark_config import LLM_MODELS
from scripts.benchmark_config import LOGS_DIR
from scripts.benchmark_config import RESULTS_DIR

logger = logging.getLogger(__name__)


def ensure_dirs() -> None:
    """Create all project directories."""
    for d in [DATA_DIR, RESULTS_DIR, CONFIGS_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)


def save_csv(df: pd.DataFrame, filename: str) -> None:
    """
    Save DataFrame to experiments/results/.

    Args:
        df: Data to save.
        filename: Output filename.
    """
    ensure_dirs()
    path = os.path.join(RESULTS_DIR, filename)
    df.to_csv(path, index=False)
    logger.info('Saved: %s', path)


def save_run_config(
    config,
    baselines_info: dict,
) -> str:
    """
    Save experiment config as JSON.

    Args:
        config: Config object.
        baselines_info: Dict with baseline stats.

    Returns:
        Path to saved config file.
    """
    ensure_dirs()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_config = {
        'timestamp': timestamp,
        'device': config.device,
        'whisper_model': config.whisper_model,
        'max_samples': config.max_samples,
        'llm_models': LLM_MODELS,
        'baselines': baselines_info,
    }
    filename = f'run_{timestamp}.json'
    path = os.path.join(CONFIGS_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(
            run_config, f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info('Config saved: %s', path)
    return path
