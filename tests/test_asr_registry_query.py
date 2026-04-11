#!/usr/bin/env python3
"""Tests for ASR registry query helpers."""

import pytest

from src.asr.registry_query import get_metadata
from src.asr.registry_query import list_models


def test_list_all_no_filters():
    """Without filters, returns all registered models."""
    models = list_models()
    assert 'whisper_base' in models
    assert 'wav2vec2_xlsr_1b_ru' in models


def test_list_by_local_profile():
    """Local profile includes Whisper and Wav2Vec2."""
    models = list_models(profile='local')
    assert set(models) == {
        'whisper_base',
        'wav2vec2_xlsr_1b_ru',
    }


def test_list_by_language_en():
    """Only Whisper supports English."""
    models = list_models(language='en')
    assert models == ['whisper_base']


def test_list_by_language_ru():
    """Both local models support Russian."""
    models = list_models(language='ru')
    assert set(models) == {
        'whisper_base',
        'wav2vec2_xlsr_1b_ru',
    }


def test_list_combined_filters():
    """Profile and language filters combine with AND."""
    models = list_models(
        profile='local', language='en',
    )
    assert models == ['whisper_base']


def test_list_returns_sorted():
    """Result is sorted alphabetically."""
    models = list_models()
    assert models == sorted(models)


def test_get_metadata_whisper():
    """get_metadata returns the full entry for whisper_base."""
    meta = get_metadata('whisper_base')
    assert meta['family'] == 'OpenAI'
    assert meta['params'] == '74M'


def test_get_metadata_unknown_key():
    """Unknown key raises KeyError with helpful message."""
    with pytest.raises(KeyError, match='Unknown ASR'):
        get_metadata('nonexistent_model')
