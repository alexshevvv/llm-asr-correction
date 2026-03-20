#!/usr/bin/env python3
"""Tests for LLM prompt generation."""

from src.correction.prompts import get_prompts


def test_english_prompts():
    """Verify English prompt contains input text."""
    sys_prompt, usr_prompt = get_prompts('hello')
    assert 'ASR' in sys_prompt
    assert 'hello' in usr_prompt


def test_russian_prompts():
    """Verify Russian prompt contains input text."""
    sys_prompt, usr_prompt = get_prompts(
        'привет', language='ru',
    )
    assert 'ASR' in sys_prompt
    assert 'привет' in usr_prompt


def test_default_is_english():
    """Verify default language is English."""
    sys_prompt, _ = get_prompts('test')
    assert 'English' in sys_prompt