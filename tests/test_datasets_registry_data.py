#!/usr/bin/env python3
"""Tests for the datasets registry metadata."""

from src.utils.datasets_registry_data import DATASETS_REGISTRY


def test_has_librispeech():
    """Registry contains LibriSpeech test-clean."""
    assert 'librispeech_test_clean' in DATASETS_REGISTRY


def test_has_fleurs_ru():
    """Registry contains FLEURS Russian."""
    assert 'fleurs_ru' in DATASETS_REGISTRY


def test_librispeech_is_english():
    """LibriSpeech test-clean is an English dataset."""
    meta = DATASETS_REGISTRY['librispeech_test_clean']
    assert meta['language'] == 'en'


def test_fleurs_is_russian():
    """FLEURS ru_ru is a Russian dataset."""
    meta = DATASETS_REGISTRY['fleurs_ru']
    assert meta['language'] == 'ru'


def test_all_entries_have_loader_path():
    """Every registry entry has a loader_path."""
    for key, meta in DATASETS_REGISTRY.items():
        assert 'loader_path' in meta, (
            f'{key} missing loader_path'
        )


def test_all_entries_have_profiles():
    """Every entry has at least one profile."""
    for key, meta in DATASETS_REGISTRY.items():
        assert meta.get('profiles'), f'{key} has no profiles'
