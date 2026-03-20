#!/usr/bin/env python3
"""Tests for experiment configuration."""

from src.utils.config import Config


def test_default_values():
    """Verify default config values."""
    config = Config()
    assert config.whisper_model == 'base'
    assert config.max_samples == 50
    assert config.llm_temperature == 0.1


def test_device_is_valid():
    """Verify device is cuda or cpu."""
    config = Config()
    assert config.device in ('cuda', 'cpu')


def test_custom_override():
    """Verify config values can be overridden."""
    config = Config(max_samples=100)
    assert config.max_samples == 100
