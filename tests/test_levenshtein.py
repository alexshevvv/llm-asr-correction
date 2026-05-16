#!/usr/bin/env python3
"""Tests for Levenshtein distance functions."""

from src.correction.levenshtein import (
    levenshtein_distance,
    normalized_levenshtein,
)


def test_identical():
    assert levenshtein_distance('cat', 'cat') == 0


def test_one_sub():
    assert levenshtein_distance('cat', 'bat') == 1


def test_insertion():
    assert levenshtein_distance('cat', 'cats') == 1


def test_deletion():
    assert levenshtein_distance('cats', 'cat') == 1


def test_empty():
    assert levenshtein_distance('', 'abc') == 3


def test_normalized_identical():
    assert normalized_levenshtein('cat', 'cat') == 0.0


def test_normalized_different():
    result = normalized_levenshtein('cat', 'dog')
    assert result == 1.0


def test_normalized_partial():
    result = normalized_levenshtein('sat', 'sit')
    assert round(result, 2) == 0.33
