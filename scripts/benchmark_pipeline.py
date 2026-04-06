#!/usr/bin/env python3
"""LLM correction pipeline for all models."""

import logging

from src.correction.llm_client import create_client
from scripts.benchmark_config import LLM_MODELS
from scripts.benchmark_io import save_csv
from scripts.run_correction import run_llm_correction

logger = logging.getLogger(__name__)


def run_all_corrections(baselines: dict) -> dict:
    """
    Run LLM correction for all models.

    Args:
        baselines: Dict of baseline DataFrames.

    Returns:
        Dict of correction DataFrames.
    """
    client = create_client()
    all_results = {}

    for model_info in LLM_MODELS:
        model_id = model_info['id']
        model_name = model_info['name']
        logger.info('Benchmarking: %s', model_name)

        for bl_key, bl_df in baselines.items():
            lang = 'en' if 'en' in bl_key else 'ru'
            key = f'{model_name}__{bl_key}'

            result_df = run_llm_correction(
                client,
                bl_df,
                language=lang,
                model=model_id,
                model_name=model_name,
            )
            all_results[key] = result_df

            if len(result_df) > 0:
                save_csv(
                    result_df,
                    f'correction_{key}.csv',
                )

    return all_results
