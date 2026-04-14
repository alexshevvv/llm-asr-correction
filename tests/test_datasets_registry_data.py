#!/usr/bin/env python3
"""Tests for the datasets registry metadata."""

from src.utils.datasets_registry_data import DATASETS_REGISTRY


def test_registry_has_five_datasets():
    """Registry contains exactly five datasets."""
    assert len(DATASETS_REGISTRY) == 5


def test_has_librispeech_clean():
    """Registry contains LibriSpeech test-clean."""
    assert 'librispeech_test_clean' in DATASETS_REGISTRY


def test_has_librispeech_other():
    """Registry contains LibriSpeech test-other."""
    assert 'librispeech_test_other' in DATASETS_REGISTRY


def test_has_fleurs_en():
    """Registry contains FLEURS English."""
    assert 'fleurs_en' in DATASETS_REGISTRY


def test_has_fleurs_ru():
    """Registry contains FLEURS Russian."""
    assert 'fleurs_ru' in DATASETS_REGISTRY


def test_has_sova_audiobooks():
    """Registry contains SOVA audiobooks."""
    assert 'sova_audiobooks_ru' in DATASETS_REGISTRY


def test_three_english_datasets():
    """Three datasets are English."""
    en = [
        k for k, m in DATASETS_REGISTRY.items()
        if m['language'] == 'en'
    ]
    assert len(en) == 3


def test_two_russian_datasets():
    """Two datasets are Russian."""
    ru = [
        k for k, m in DATASETS_REGISTRY.items()
        if m['language'] == 'ru'
    ]
    assert len(ru) == 2


def test_librispeech_other_is_noisy():
    """LibriSpeech test-other has 'noisy' style."""
    meta = DATASETS_REGISTRY['librispeech_test_other']
    assert meta['style'] == 'noisy'


def test_sova_is_literary():
    """SOVA audiobooks has 'literary' style."""
    meta = DATASETS_REGISTRY['sova_audiobooks_ru']
    assert meta['style'] == 'literary'


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
