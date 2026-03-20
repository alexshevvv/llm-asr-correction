#!/usr/bin/env python3
"""Tests for audio utilities."""

import os

import numpy as np

from src.utils.audio import save_temp_wav


def test_save_creates_file():
    """Verify temp WAV file is created."""
    audio = np.zeros(16000, dtype=np.float32)
    path = save_temp_wav(audio, 16000)

    assert os.path.exists(path)
    assert path.endswith('.wav')

    os.unlink(path)


def test_save_file_not_empty():
    """Verify saved WAV file has content."""
    audio = np.zeros(16000, dtype=np.float32)
    path = save_temp_wav(audio, 16000)

    assert os.path.getsize(path) > 0

    os.unlink(path)
