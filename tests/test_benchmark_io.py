#!/usr/bin/env python3
"""Tests for benchmark I/O utilities."""

import pandas as pd

from scripts.benchmark_io import ensure_dirs
from scripts.benchmark_io import save_csv


def test_create_dir(tmp_path, monkeypatch):
    """All project directories are created."""
    monkeypatch.setattr(
        'scripts.benchmark_io.DATA_DIR',
        str(tmp_path / 'data'),
    )
    monkeypatch.setattr(
        'scripts.benchmark_io.RESULTS_DIR',
        str(tmp_path / 'results'),
    )
    monkeypatch.setattr(
        'scripts.benchmark_io.CONFIGS_DIR',
        str(tmp_path / 'configs'),
    )
    monkeypatch.setattr(
        'scripts.benchmark_io.LOGS_DIR',
        str(tmp_path / 'logs'),
    )
    ensure_dirs()
    assert (tmp_path / 'data').exists()
    assert (tmp_path / 'results').exists()
    assert (tmp_path / 'configs').exists()
    assert (tmp_path / 'logs').exists()


def test_writes_file(tmp_path, monkeypatch):
    """CSV file is written to results dir."""
    monkeypatch.setattr(
        'scripts.benchmark_io.RESULTS_DIR',
        str(tmp_path),
    )
    df = pd.DataFrame({'col': [1, 2, 3]})
    save_csv(df, 'test.csv')
    path = tmp_path / 'test.csv'
    assert path.exists()
    loaded = pd.read_csv(path)
    assert len(loaded) == 3
