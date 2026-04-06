#!/usr/bin/env python3
"""Tests for benchmark summary builder."""

import pytest

import pandas as pd

from scripts.benchmark_summary import build_summary


def _make_result_df(
    bl_wer: float = 0.2,
    cr_wer: float = 0.1,
) -> pd.DataFrame:
    """Create mock correction result."""
    return pd.DataFrame([{
        'wer_baseline': bl_wer,
        'wer_corrected': cr_wer,
        'wer_improved': bl_wer > cr_wer,
        'wer_degraded': bl_wer < cr_wer,
    }])


def test_build_summary_single_experiment():
    """Summary has correct WER change."""
    results = {
        'TestLLM__whisper_en': _make_result_df(0.2, 0.1),
    }
    summary = build_summary(results)
    assert len(summary) == 1
    assert summary.iloc[0]['LLM'] == 'TestLLM'
    assert summary.iloc[0]['ASR'] == 'whisper_en'
    assert summary.iloc[0]['WER Change (%)'] == pytest.approx(50.0)


def test_build_summary_skips_empty():
    """Empty DataFrames are skipped."""
    results = {
        'LLM__asr': pd.DataFrame(),
        'LLM2__asr': _make_result_df(),
    }
    summary = build_summary(results)
    assert len(summary) == 1


def test_build_summary_degraded():
    """Degradation shows negative change."""
    results = {
        'LLM__asr': _make_result_df(0.1, 0.2),
    }
    summary = build_summary(results)
    assert summary.iloc[0]['WER Change (%)'] < 0
