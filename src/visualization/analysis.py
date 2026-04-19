#!/usr/bin/env python3
"""Build unified analysis DataFrame from benchmark results."""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def build_analysis_df(
    results: dict[str, pd.DataFrame],
    asr_registry: dict[str, dict[str, Any]],
    datasets_registry: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """
    Assemble a unified DataFrame from benchmark result dict.

    """

    rows = []
    for key, df in results.items():
        if len(df) == 0:
            continue
        parts = key.split('__')
        if len(parts) != 3:
            logger.warning('Skipping malformed key: %s', key)
            continue
        llm_name, asr_key, dataset_key = parts
        asr_meta = asr_registry.get(asr_key, {})
        ds_meta = datasets_registry.get(dataset_key, {})

        bl_wer = df['wer_baseline'].mean()
        cr_wer = df['wer_corrected'].mean()
        change_pct = (
            (bl_wer - cr_wer) / bl_wer * 100
            if bl_wer > 0 else 0.0
        )

        asr_display = asr_meta.get('display_name', asr_key)
        ds_language = ds_meta.get('language', '?')
        ds_display = ds_meta.get('display_name', dataset_key)
        experiment = f'{asr_display} / {ds_display}'

        rows.append({
            'LLM': llm_name,
            'ASR': asr_display,
            'ASR_key': asr_key,
            'Dataset': ds_meta.get('display_name', dataset_key),
            'Dataset_key': dataset_key,
            'Experiment': experiment,
            'Language': ds_language,
            'Samples': len(df),
            'Baseline WER': bl_wer,
            'Corrected WER': cr_wer,
            'WER Change (%)': change_pct,
            'Improved': int(df['wer_improved'].sum()),
            'Degraded': int(df['wer_degraded'].sum()),
            'Unchanged': int(
                (
                    ~df['wer_improved']
                    & ~df['wer_degraded']
                ).sum()
            ),
        })

    return pd.DataFrame(rows)
