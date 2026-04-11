#!/usr/bin/env python3
"""Tests for memory management helpers."""

from src.utils.memory import release_model
from src.utils.memory import report_memory


class FakeModel:
    """Minimal stand-in for a torch model."""

    def __init__(self):
        self.weights = [1, 2, 3]
        self.processor = 'fake'


def test_report_memory_returns_dict():
    """report_memory returns a dict with expected keys."""
    mem = report_memory()
    assert 'allocated_gb' in mem
    assert 'reserved_gb' in mem
    assert 'total_gb' in mem


def test_report_memory_values_non_negative():
    """All reported memory values are non-negative."""
    mem = report_memory()
    assert mem['allocated_gb'] >= 0.0
    assert mem['reserved_gb'] >= 0.0
    assert mem['total_gb'] >= 0.0


def test_release_model_clears_attributes():
    """release_model strips attributes from the object."""
    model = FakeModel()
    assert hasattr(model, 'weights')
    release_model(model)
    assert not hasattr(model, 'weights')
    assert not hasattr(model, 'processor')


def test_release_model_safe_on_empty_object():
    """release_model does not raise on empty object."""
    class Empty:
        pass
    release_model(Empty())
