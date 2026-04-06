#!/usr/bin/env python3
"""Build benchmark summary table."""

import pandas as pd


def build_summary(results: dict) -> pd.DataFrame:
    """
    Build summary table from all results.

    Args:
        results: Dict of correction DataFrames.

    Returns:
        Summary DataFrame with WER changes.
    """
    rows = []
    for key, df in results.items():
        if len(df) == 0:
            continue
        llm_name, asr_key = key.split('__')
        bl_wer = df['wer_baseline'].mean()
        cr_wer = df['wer_corrected'].mean()
        rows.append({
            'LLM': llm_name,
            'ASR': asr_key,
            'Samples': len(df),
            'Baseline WER': bl_wer,
            'Corrected WER': cr_wer,
            'WER Change (%)': (
                (bl_wer - cr_wer) / bl_wer * 100
            ),
            'Improved': int(df['wer_improved'].sum()),
            'Degraded': int(df['wer_degraded'].sum()),
        })
    return pd.DataFrame(rows)
