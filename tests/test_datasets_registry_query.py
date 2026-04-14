#!/usr/bin/env python3
"""Tests for datasets registry query helpers."""

import pytest

from src.utils.datasets_registry_query import (
    get_dataset_metadata,
)
from src.utils.datasets_registry_query import list_datasets


def test_list_all():
    """Without filters, returns all five datasets."""
    keys = list_datasets()
    assert set(keys) == {
        'fleurs_en',
        'fleurs_ru',
        'librispeech_test_clean',
        'librispeech_test_other',
        'sova_audiobooks_ru',
    }


def test_list_english_only():
    """English filter returns three EN datasets."""
    keys = list_datasets(language='en')
    assert set(keys) == {
        'fleurs_en',
        'librispeech_test_clean',
        'librispeech_test_other',
    }


def test_list_russian_only():
    """Russian filter returns two RU datasets."""
    keys = list_datasets(language='ru')
    assert set(keys) == {
        'fleurs_ru',
        'sova_audiobooks_ru',
    }


def test_list_local_profile():
    """Local profile includes all five datasets."""
    keys = list_datasets(profile='local')
    assert 'librispeech_test_clean' in keys
    assert 'librispeech_test_other' in keys
    assert 'fleurs_en' in keys
    assert 'fleurs_ru' in keys
    assert 'sova_audiobooks_ru' in keys


def test_list_returns_sorted():
    """Result is sorted alphabetically."""
    keys = list_datasets()
    assert keys == sorted(keys)


def test_get_metadata_returns_dict():
    """get_dataset_metadata returns the full entry."""
    meta = get_dataset_metadata('fleurs_ru')
    assert meta['source'] == 'Google FLEURS'


def test_get_metadata_unknown_raises():
    """Unknown key raises KeyError."""
    with pytest.raises(KeyError, match='Unknown dataset'):
        get_dataset_metadata('nonexistent')
