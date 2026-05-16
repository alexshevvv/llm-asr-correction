#!/usr/bin/env python3
"""Tests for phonetic filter."""

from src.correction.phonetic_filter import (
    apply_phonetic_filter,
)


def test_no_changes():
    text = 'the cat sat on the mat'
    filtered, stats = apply_phonetic_filter(text, text)
    assert filtered == text
    assert stats['reverted_subs'] == 0


def test_close_sub_accepted():
    filtered, stats = apply_phonetic_filter(
        'the cat set on the mat',
        'the cat sat on the mat',
        threshold=0.5,
    )
    assert 'sat' in filtered
    assert stats['accepted_subs'] >= 1


def test_far_sub_reverted():
    filtered, stats = apply_phonetic_filter(
        'the cat set on the mat',
        'the cat dog on a mat',
        threshold=0.5,
    )
    assert stats['reverted_subs'] >= 1


def test_deletion_reverted():
    filtered, stats = apply_phonetic_filter(
        'the cat sat on the mat',
        'the cat sat on mat',
        threshold=0.5,
    )
    assert stats['reverted_dels'] >= 1


def test_empty_input():
    filtered, stats = apply_phonetic_filter('', 'hello')
    assert stats['reverted_subs'] == 0
