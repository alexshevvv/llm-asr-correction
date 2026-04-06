#!/usr/bin/env python3
"""Tests for visualization common utilities."""

import pandas as pd

from src.visualization.common import build_viz_df


def _make_correction_df(
    bl_wer: float = 0.2,
    cr_wer: float = 0.15,
) -> pd.DataFrame:
    """Create mock correction DataFrame."""
    return pd.DataFrame([{
        'wer_baseline': bl_wer,
        'wer_corrected': cr_wer,
        'wer_improved': True,
        'wer_degraded': False,
    }])


def test_build_viz_df_basic():
    """Builds correct visualization DataFrame."""
    results = {
        'MyLLM__whisper_en': _make_correction_df(),
    }
    viz_df = build_viz_df(results)
    assert len(viz_df) == 1
    assert viz_df.iloc[0]['LLM'] == 'MyLLM'
    assert viz_df.iloc[0]['ASR'] == 'whisper_en'


def test_build_viz_df_wer():
    """Improvement shows positive WER change."""
    results = {
        'LLM__asr': _make_correction_df(0.2, 0.1),
    }
    viz_df = build_viz_df(results)
    assert viz_df.iloc[0]['WER Change (%)'] > 0


def test_build_viz_df_skip_empty():
    """Empty DataFrames are excluded."""
    results = {
        'LLM__asr1': pd.DataFrame(),
        'LLM__asr2': _make_correction_df(),
    }
    viz_df = build_viz_df(results)
    assert len(viz_df) == 1
