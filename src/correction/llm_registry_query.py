#!/usr/bin/env python3
"""Query helpers for the LLM registry."""

from typing import Any

from src.correction.llm_registry_data import LLM_REGISTRY


def list_llm_models(
    family: str | None = None,
    architecture: str | None = None,
) -> list[str]:
    """
    List LLM model keys matching filters.

    Args:
        family: Filter by developer
            (Alibaba, DeepSeek, OpenAI, Meta, etc.).
        architecture: Filter by architecture type
            (dense or MoE).

    Returns:
        Sorted list of registry keys.
    """

    keys = []
    for key, meta in LLM_REGISTRY.items():
        if family and meta['family'] != family:
            continue
        if architecture and meta['architecture'] != architecture:
            continue
        keys.append(key)
    return sorted(keys)


def get_llm_metadata(key: str) -> dict[str, Any]:
    """
    Return metadata dict for an LLM key.

    Args:
        key: Registry key.

    Returns:
        Metadata dictionary.

    Raises:
        KeyError: If key is not in the registry.
    """

    if key not in LLM_REGISTRY:
        raise KeyError(
            f'Unknown LLM key: {key}. '
            f'Available: {sorted(LLM_REGISTRY.keys())}'
        )
    return LLM_REGISTRY[key]
