#!/usr/bin/env python3
"""Tests for dotted-path class resolver."""

import pytest

from src.utils.class_loader import resolve_class


def test_resolve_base_asr():
    """Resolve the BaseASR abstract class."""
    cls = resolve_class('src.asr.base.BaseASR')
    assert cls.__name__ == 'BaseASR'


def test_resolve_whisper_asr():
    """Resolve the concrete WhisperASR class."""
    cls = resolve_class(
        'src.asr.whisper_transcribe.WhisperASR',
    )
    assert cls.__name__ == 'WhisperASR'


def test_unknown_module_raises():
    """Unknown module path raises ImportError."""
    with pytest.raises(ImportError):
        resolve_class('src.nonexistent.module.Class')


def test_unknown_class_raises():
    """Unknown class in existing module raises AttributeError."""
    with pytest.raises(AttributeError):
        resolve_class('src.asr.base.NoSuchClass')
