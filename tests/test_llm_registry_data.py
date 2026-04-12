#!/usr/bin/env python3
"""Tests for the LLM registry metadata."""

from src.correction.llm_registry_data import LLM_REGISTRY


def test_registry_has_six_models():
    """Registry contains exactly six LLMs."""
    assert len(LLM_REGISTRY) == 6


def test_registry_has_llama():
    """Registry contains Llama 3.3 70B."""
    assert 'llama_3_3_70b' in LLM_REGISTRY


def test_registry_has_deepseek():
    """Registry contains DeepSeek V3."""
    assert 'deepseek_v3' in LLM_REGISTRY


def test_all_entries_have_hf_id():
    """Every entry has a HuggingFace model id."""
    for key, meta in LLM_REGISTRY.items():
        assert 'hf_id' in meta, f'{key} missing hf_id'


def test_all_entries_have_family():
    """Every entry has a developer family."""
    for key, meta in LLM_REGISTRY.items():
        assert meta.get('family'), f'{key} has no family'


def test_architecture_is_valid():
    """Architecture is either 'dense' or 'MoE'."""
    for key, meta in LLM_REGISTRY.items():
        assert meta['architecture'] in ('dense', 'MoE'), (
            f'{key} has invalid architecture'
        )


def test_three_alibaba_models():
    """Three models come from Alibaba (Qwen family)."""
    alibaba = [
        k for k, m in LLM_REGISTRY.items()
        if m['family'] == 'Alibaba'
    ]
    assert len(alibaba) == 3
