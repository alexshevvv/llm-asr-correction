#!/usr/bin/env python3
"""Tests for dataset loader resolver."""

import pytest

from src.utils.dataset_loader import load_dataset_by_key


def test_unknown_key_raises():
    """Unknown dataset key raises KeyError."""
    with pytest.raises(KeyError, match='Unknown dataset'):
        load_dataset_by_key('nonexistent', max_samples=1)
