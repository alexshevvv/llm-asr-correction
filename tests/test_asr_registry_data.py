#!/usr/bin/env python3
"""Tests for the ASR registry metadata."""

from src.asr.registry_data import ASR_REGISTRY


def test_registry_has_whisper():
    """Registry contains Whisper base."""
    assert 'whisper_base' in ASR_REGISTRY


def test_registry_has_wav2vec2():
    """Registry contains Wav2Vec2 XLS-R."""
    assert 'wav2vec2_xlsr_1b_ru' in ASR_REGISTRY


def test_whisper_supports_both_languages():
    """Whisper supports English and Russian."""
    meta = ASR_REGISTRY['whisper_base']
    assert set(meta['languages']) == {'en', 'ru'}


def test_wav2vec2_russian_only():
    """Wav2Vec2 XLS-R supports only Russian."""
    meta = ASR_REGISTRY['wav2vec2_xlsr_1b_ru']
    assert meta['languages'] == ['ru']


def test_all_entries_have_class_path():
    """Every registry entry has class_path."""
    for key, meta in ASR_REGISTRY.items():
        assert 'class_path' in meta, (
            f'{key} missing class_path'
        )


def test_all_entries_have_profiles():
    """Every entry has at least one profile."""
    for key, meta in ASR_REGISTRY.items():
        assert meta.get('profiles'), (
            f'{key} has no profiles'
        )
