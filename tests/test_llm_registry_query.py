#!/usr/bin/env python3
"""Tests for the LLM registry query helpers."""

import pytest

from src.correction.llm_registry_query import get_llm_metadata
from src.correction.llm_registry_query import list_llm_models


def test_list_all_no_filters():
    """list_llm_models() returns all six models."""
    models = list_llm_models()
    assert len(models) == 6


def test_list_by_family_alibaba():
    """Alibaba family filter returns three Qwen models."""
    models = list_llm_models(family='Alibaba')
    assert set(models) == {
        'qwen2_5_7b', 'qwen2_5_72b', 'qwen3_235b',
    }


def test_list_by_family_meta():
    """Meta family filter returns Llama only."""
    models = list_llm_models(family='Meta')
    assert models == ['llama_3_3_70b']


def test_list_by_architecture_dense():
    """Dense filter returns three dense LLMs."""
    models = list_llm_models(architecture='dense')
    assert set(models) == {
        'qwen2_5_7b', 'qwen2_5_72b', 'llama_3_3_70b',
    }


def test_list_by_architecture_moe():
    """MoE filter returns three MoE LLMs."""
    models = list_llm_models(architecture='MoE')
    assert set(models) == {
        'qwen3_235b', 'deepseek_v3', 'gpt_oss_120b',
    }


def test_list_combined_filters():
    """Combined filters (family + architecture) use AND."""
    models = list_llm_models(
        family='Alibaba', architecture='MoE',
    )
    assert models == ['qwen3_235b']


def test_list_returns_sorted():
    """Result is sorted alphabetically."""
    models = list_llm_models()
    assert models == sorted(models)


def test_get_metadata_llama():
    """get_llm_metadata returns full entry for Llama."""
    meta = get_llm_metadata('llama_3_3_70b')
    assert meta['family'] == 'Meta'
    assert meta['hf_id'] == 'meta-llama/Llama-3.3-70B-Instruct'


def test_get_metadata_unknown_raises():
    """Unknown key raises KeyError with helpful message."""
    with pytest.raises(KeyError, match='Unknown LLM'):
        get_llm_metadata('nonexistent_llm')
