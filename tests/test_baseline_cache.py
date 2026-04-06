#!/usr/bin/env python3
"""Tests for baseline caching utilities."""

import pandas as pd

from scripts.baseline_cache import load_cached
from scripts.baseline_cache import save_cache


def test_load_cached_none():
    """Returns None when file does not exist."""
    result = load_cached('nonexistent_file.csv')
    assert result is None


def test_save_load_cache(tmp_path, monkeypatch):
    """Saved cache can be loaded back."""
    monkeypatch.setattr(
        'scripts.baseline_cache.CACHE_DIR',
        str(tmp_path),
    )
    df = pd.DataFrame({
        'id': [1, 2],
        'wer': [0.1, 0.2],
    })
    save_cache(df, 'test_baseline.csv')
    loaded = load_cached('test_baseline.csv')
    assert loaded is not None
    assert len(loaded) == 2


def test_cache_create_dir(tmp_path, monkeypatch):
    """Cache dir is created if missing."""
    cache_dir = tmp_path / 'new_dir'
    monkeypatch.setattr(
        'scripts.baseline_cache.CACHE_DIR',
        str(cache_dir),
    )
    df = pd.DataFrame({'x': [1]})
    save_cache(df, 'test.csv')
    assert cache_dir.exists()
