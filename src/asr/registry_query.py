#!/usr/bin/env python3
"""Query helpers for the ASR registry."""

from typing import Any

from src.asr.registry_data import ASR_REGISTRY


def list_models(
    profile: str | None = None,
    language: str | None = None,
) -> list[str]:
    """
    List ASR model keys matching filters.

    Args:
        profile: Filter by execution profile.
        language: Filter by supported language.

    Returns:
        Sorted list of registry keys.
    """

    keys = []
    for key, meta in ASR_REGISTRY.items():
        if profile and profile not in meta['profiles']:
            continue
        if language and language not in meta['languages']:
            continue
        keys.append(key)
    return sorted(keys)


def get_metadata(key: str) -> dict[str, Any]:
    """
    Return metadata dict for a model key.

    Args:
        key: Registry key.

    Returns:
        Metadata dictionary.

    Raises:
        KeyError: If key is not in the registry.
    """

    if key not in ASR_REGISTRY:
        raise KeyError(
            f'Unknown ASR model key: {key}. '
            f'Available: {sorted(ASR_REGISTRY.keys())}'
        )
    return ASR_REGISTRY[key]
