#!/usr/bin/env python3
"""Tests for word-level error classification."""

from src.evaluation.error_classification import (
    classify_errors,
)


def test_identical_strings():
    """No errors when texts match."""
    result = classify_errors('the cat sat', 'the cat sat')
    assert result['substitutions'] == 0
    assert result['insertions'] == 0
    assert result['deletions'] == 0
    assert result['hits'] == 3


def test_substitution():
    """One word replaced."""
    result = classify_errors('the cat sat', 'the dog sat')
    assert result['substitutions'] == 1
    assert result['insertions'] == 0
    assert result['deletions'] == 0


def test_insertion():
    """Extra word added."""
    result = classify_errors('the cat', 'the big cat')
    assert result['insertions'] == 1


def test_deletion():
    """Word removed."""
    result = classify_errors('the big cat', 'the cat')
    assert result['deletions'] == 1


def test_empty_reference():
    """Empty reference counts all hyp words as insertions."""
    result = classify_errors('', 'hello world')
    assert result['insertions'] == 2
    assert result['ref_words'] == 0


def test_empty_both():
    """Both empty gives zero errors."""
    result = classify_errors('', '')
    assert result['substitutions'] == 0
    assert result['insertions'] == 0
    assert result['deletions'] == 0
