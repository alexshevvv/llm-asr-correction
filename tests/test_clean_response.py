#!/usr/bin/env python3
"""Tests for LLM response cleaning."""

from src.correction.clean_response import clean_response


def test_empty_response():
    """Empty response returns original."""
    assert clean_response('', 'hello') == 'hello'


def test_returns_normal():
    """Normal correction passes through."""
    assert clean_response('fixed text', 'orig') == 'fixed text'


def test_truncates_long():
    """Long response truncated to first line."""
    long = 'Fixed line\nHere is explanation...' + 'x' * 500
    result = clean_response(long, 'short')
    assert result == 'Fixed line'


def test_removes_prefix():
    """Common prefixes are stripped."""
    assert clean_response(
        'Corrected: hello world', 'helo wrld',
    ) == 'hello world'


def test_removes_quotes():
    """Surrounding quotes are removed."""
    assert clean_response(
        '"hello world"', 'helo',
    ) == 'hello world'
