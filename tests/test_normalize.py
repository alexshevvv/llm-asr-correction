#!/usr/bin/env python3
"""Tests for text normalization."""

from src.evaluation.normalize import normalize_text


def test_lowercase():
    """Verify text is lowercased."""
    assert normalize_text('Hello World') == 'hello world'


def test_remove_punctuation():
    """Verify punctuation is removed."""
    assert normalize_text('Hello, World!') == 'hello world'


def test_collapse_spaces():
    """Verify multiple spaces collapse to one."""
    assert normalize_text('  hello   world  ') == 'hello world'


def test_keep_case_when_disabled():
    """Verify case is preserved when disabled."""
    result = normalize_text('Hello', lowercase=False)
    assert result == 'Hello'


def test_empty_string():
    """Verify empty input returns empty string."""
    assert normalize_text('') == ''
