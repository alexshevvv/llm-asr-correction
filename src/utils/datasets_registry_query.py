#!/usr/bin/env python3
"""Query helpers for the datasets registry."""

from typing import Any

from src.utils.datasets_registry_data import DATASETS_REGISTRY


def list_datasets(
    profile: str | None = None,
    language: str | None = None,
) -> list[str]:
    """
    List dataset keys matching filters.

    Args:
        profile: Filter by execution profile.
        language: Filter by language (en or ru).

    Returns:
        Sorted list of registry keys.
    """

    keys = []
    for key, meta in DATASETS_REGISTRY.items():
        if profile and profile not in meta['profiles']:
            continue
        if language and meta['language'] != language:
            continue
        keys.append(key)
    return sorted(keys)


def get_dataset_metadata(key: str) -> dict[str, Any]:
    """
    Return metadata dict for a dataset key.

    Args:
        key: Registry key.

    Returns:
        Metadata dictionary.

    Raises:
        KeyError: If key is not in the registry.
    """

    if key not in DATASETS_REGISTRY:
        raise KeyError(
            f'Unknown dataset key: {key}. '
            f'Available: {sorted(DATASETS_REGISTRY.keys())}'
        )
    return DATASETS_REGISTRY[key]
