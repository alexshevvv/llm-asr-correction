#!/usr/bin/env python3
"""Tests for the analysis DataFrame builder."""

import pandas as pd

from src.visualization.analysis import build_analysis_df


FAKE_ASR = {
    'whisper_base': {
        'display_name': 'Whisper base',
        'languages': ['en', 'ru'],
    },
}

FAKE_DS = {
    'librispeech_test_clean': {
        'display_name': 'LibriSpeech test-clean',
        'language': 'en',
    },
}


def _fake_result_df():
    """Build a tiny result DataFrame for one experiment."""
    return pd.DataFrame([
        {
            'wer_baseline': 0.10,
            'wer_corrected': 0.05,
            'wer_improved': True,
            'wer_degraded': False,
        },
        {
            'wer_baseline': 0.20,
            'wer_corrected': 0.25,
            'wer_improved': False,
            'wer_degraded': True,
        },
    ])


def test_build_df_one_experiment():
    """Single experiment builds a one-row DataFrame."""
    results = {
        'Qwen2.5 7B__whisper_base__librispeech_test_clean':
            _fake_result_df(),
    }
    df = build_analysis_df(results, FAKE_ASR, FAKE_DS)
    assert len(df) == 1


def test_build_df_columns_present():
    """Expected columns are in the output."""
    results = {
        'Qwen2.5 7B__whisper_base__librispeech_test_clean':
            _fake_result_df(),
    }
    df = build_analysis_df(results, FAKE_ASR, FAKE_DS)
    for col in [
        'LLM', 'ASR', 'Experiment',
        'Baseline WER', 'Corrected WER',
    ]:
        assert col in df.columns


def test_build_df_skips_empty():
    """Empty DataFrame input is skipped."""
    results = {
        'Qwen2.5 7B__whisper_base__librispeech_test_clean':
            pd.DataFrame(),
    }
    df = build_analysis_df(results, FAKE_ASR, FAKE_DS)
    assert len(df) == 0


def test_build_df_skips_malformed_key():
    """Keys that do not split into 3 parts are skipped."""
    results = {
        'bad_key_format': _fake_result_df(),
    }
    df = build_analysis_df(results, FAKE_ASR, FAKE_DS)
    assert len(df) == 0


def test_build_df_computes_change_percent():
    """WER change percentage is computed correctly."""
    results = {
        'Qwen__whisper_base__librispeech_test_clean':
            _fake_result_df(),
    }
    df = build_analysis_df(results, FAKE_ASR, FAKE_DS)
    bl = df['Baseline WER'].iloc[0]
    cr = df['Corrected WER'].iloc[0]
    expected = (bl - cr) / bl * 100
    assert abs(df['WER Change (%)'].iloc[0] - expected) < 1e-6
