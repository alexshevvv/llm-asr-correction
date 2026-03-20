#!/usr/bin/env python3
"""Tests for WER and CER metrics."""

from src.evaluation.metrics import calculate_cer
from src.evaluation.metrics import calculate_wer


def test_wer_perfect():
    """Verify WER is 0 for identical texts."""
    assert calculate_wer('hello world', 'hello world') == 0.0


def test_wer_one_substitution():
    """Verify WER for a single word substitution."""
    result = calculate_wer(
        'the cat sat on the mat',
        'the cat set on the mat',
    )
    assert abs(result - 1 / 6) < 0.01


def test_wer_empty_reference():
    """Verify WER is 1.0 when reference is empty."""
    assert calculate_wer('', 'some text') == 1.0


def test_wer_both_empty():
    """Verify WER is 0 when both are empty."""
    assert calculate_wer('', '') == 0.0


def test_cer_perfect():
    """Verify CER is 0 for identical texts."""
    assert calculate_cer('hello', 'hello') == 0.0


def test_cer_one_char_error():
    """Verify CER for a single character error."""
    assert calculate_cer('hello', 'hallo') > 0.0
