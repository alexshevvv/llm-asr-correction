#!/usr/bin/env python3
"""Run LLM correction experiments."""

import logging

import pandas as pd
from tqdm.auto import tqdm

from src.correction.llm_client import correct_with_llm
from src.evaluation import calculate_cer
from src.evaluation import calculate_wer

logger = logging.getLogger(__name__)


def run_llm_correction(
    client,
    baseline_df: pd.DataFrame,
    language: str = 'en',
    model: str = 'Qwen/Qwen2.5-72B-Instruct',
    model_name: str = '',
) -> pd.DataFrame:
    """
    Apply LLM correction to ASR outputs.

    Args:
        client: HF Inference API client.
        baseline_df: DataFrame with baseline results.
        language: Language code (en or ru).
        model: HuggingFace model identifier.
        model_name: Display name for results.

    Returns:
        DataFrame with correction results.
    """
    errors_df = baseline_df[
        baseline_df['wer'] > 0
    ].copy()

    if len(errors_df) == 0:
        logger.info('No errors to correct')
        return pd.DataFrame()

    label = model_name or model
    results = []
    desc = f'{label} ({language})'

    for _, row in tqdm(
        errors_df.iterrows(),
        total=len(errors_df),
        desc=desc,
    ):
        corrected = correct_with_llm(
            client,
            row['hypothesis'],
            language=language,
            model=model,
        )
        new_wer = calculate_wer(
            row['reference'], corrected,
        )
        new_cer = calculate_cer(
            row['reference'], corrected,
        )

        results.append({
            'id': row['id'],
            'reference': row['reference'],
            'asr_output': row['hypothesis'],
            'llm_corrected': corrected,
            'wer_baseline': row['wer'],
            'wer_corrected': new_wer,
            'cer_baseline': row['cer'],
            'cer_corrected': new_cer,
            'wer_improved': row['wer'] > new_wer,
            'wer_degraded': row['wer'] < new_wer,
            'llm_model': label,
            'model': row['model'],
            'language': language,
        })

    return pd.DataFrame(results)
