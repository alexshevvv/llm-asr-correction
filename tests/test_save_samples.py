#!/usr/bin/env python3
"""Tests for audio sample saving utility."""

import numpy as np

from src.utils.save_samples import save_audio_samples


def _make_samples(count: int = 3) -> list[dict]:
    """Create mock audio samples."""
    return [
        {
            'audio': np.random.randn(16000).astype(
                np.float32,
            ),
            'sample_rate': 16000,
            'reference': f'sample {i}',
            'id': i,
        }
        for i in range(count)
    ]


def test_saves_wav_files(tmp_path, monkeypatch):
    """WAV files are created in samples dir."""
    monkeypatch.setattr(
        'src.utils.save_samples.SAMPLES_DIR',
        str(tmp_path),
    )
    samples = _make_samples(3)
    save_audio_samples(samples, prefix='test')
    assert (tmp_path / 'test_0.wav').exists()
    assert (tmp_path / 'test_1.wav').exists()
    assert (tmp_path / 'test_2.wav').exists()


def test_respects_count(tmp_path, monkeypatch):
    """Only requested number of samples saved."""
    monkeypatch.setattr(
        'src.utils.save_samples.SAMPLES_DIR',
        str(tmp_path),
    )
    samples = _make_samples(5)
    save_audio_samples(samples, prefix='x', count=2)
    assert (tmp_path / 'x_0.wav').exists()
    assert (tmp_path / 'x_1.wav').exists()
    assert not (tmp_path / 'x_2.wav').exists()


def test_skips_existing(tmp_path, monkeypatch):
    """Existing files are not overwritten."""
    monkeypatch.setattr(
        'src.utils.save_samples.SAMPLES_DIR',
        str(tmp_path),
    )
    existing = tmp_path / 'ru_0.wav'
    existing.write_text('existing')
    original_size = existing.stat().st_size

    samples = _make_samples(1)
    save_audio_samples(samples, prefix='ru')
    assert existing.stat().st_size == original_size
