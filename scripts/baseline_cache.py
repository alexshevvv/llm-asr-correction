#!/usr/bin/env python3
"""Baseline caching utilities."""

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join('data', 'processed')


def load_cached(filename: str) -> pd.DataFrame | None:
    """
    Load cached baseline if exists.

    Args:
        filename: CSV filename in cache dir.

    Returns:
        DataFrame or None if not cached.
    """
    path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(path):
        logger.info('Loaded cached: %s', path)
        return pd.read_csv(path)
    return None


def save_cache(
    df: pd.DataFrame, filename: str,
) -> None:
    """
    Save baseline to cache.

    Args:
        df: Baseline DataFrame.
        filename: CSV filename.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, filename)
    df.to_csv(path, index=False)
    logger.info('Cached: %s', path)
