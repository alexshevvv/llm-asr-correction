#!/usr/bin/env python3
"""Resolve and invoke dataset loaders from the registry."""

import logging
from typing import Any

from src.utils.class_loader import resolve_class
from src.utils.datasets_registry_query import (
    get_dataset_metadata,
)

logger = logging.getLogger(__name__)


def load_dataset_by_key(
    key: str, max_samples: int = 50,
) -> list[dict[str, Any]]:
    """
    Resolve and call the loader function for a dataset key.

    Args:
        key: Registry key (e.g. 'librispeech_test_clean').
        max_samples: Number of samples to load.

    Returns:
        List of sample dicts from the loader.
    """

    meta = get_dataset_metadata(key)
    loader = resolve_class(meta['loader_path'])
    logger.info(
        'Loading %s (%d samples)',
        meta.get('display_name', key), max_samples,
    )
    return loader(max_samples=max_samples)
