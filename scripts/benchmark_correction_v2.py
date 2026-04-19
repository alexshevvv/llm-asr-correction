#!/usr/bin/env python3
"""Registry-driven LLM correction for baseline results."""

import logging

import pandas as pd
from openai import OpenAI

from src.correction.llm_client import correct_with_llm
from src.correction.llm_registry_data import LLM_REGISTRY
from src.correction.llm_registry_query import list_llm_models
from src.evaluation.metrics import calculate_wer
from src.utils.datasets_registry_data import DATASETS_REGISTRY

logger = logging.getLogger(__name__)


def run_correction_matrix(
    baselines: dict[str, pd.DataFrame],
    client: OpenAI,
) -> dict[str, pd.DataFrame]:
    """
    Run LLM correction over every (LLM × baseline experiment).

    Args:
        baselines: Dict '<asr>__<dataset>' -> baseline df
            with 'reference', 'hypothesis', 'wer' columns.
        client: HF Inference API client.

    Returns:
        Dict '<llm>__<asr>__<dataset>' -> correction DataFrame
        with columns 'reference', 'hypothesis', 'corrected',
        'wer_baseline', 'wer_corrected',
        'wer_improved', 'wer_degraded'.
    """

    results = {}
    llm_keys = list_llm_models()
    total = len(llm_keys) * len(baselines)
    logger.info('Correction matrix: %d experiments', total)

    for llm_key in llm_keys:
        llm_meta = LLM_REGISTRY[llm_key]
        hf_id = llm_meta['hf_id']
        llm_name = llm_meta['display_name']

        for bl_key, bl_df in baselines.items():
            ds_key = bl_key.split('__')[1]
            ds_meta = DATASETS_REGISTRY.get(ds_key, {})
            lang = ds_meta.get('language', 'en')
            rows = []
            for _, r in bl_df.iterrows():
                corrected = correct_with_llm(
                    client=client,
                    text=r['hypothesis'],
                    language=lang,
                    model=hf_id,
                )
                wer_c = calculate_wer(r['reference'], corrected)
                rows.append({
                    'reference': r['reference'],
                    'hypothesis': r['hypothesis'],
                    'corrected': corrected,
                    'wer_baseline': r['wer'],
                    'wer_corrected': wer_c,
                    'wer_improved': wer_c < r['wer'],
                    'wer_degraded': wer_c > r['wer'],
                })

            df = pd.DataFrame(rows)
            key = f'{llm_name}__{bl_key}'
            results[key] = df

            bl_wer = df['wer_baseline'].mean()
            cr_wer = df['wer_corrected'].mean()
            logger.info(
                '%s: %.2f%% -> %.2f%%',
                key, bl_wer * 100, cr_wer * 100,
            )

    return results
