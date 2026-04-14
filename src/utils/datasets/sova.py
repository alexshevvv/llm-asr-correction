#!/usr/bin/env python3
"""SOVA audiobook dataset loader."""

import logging
import os

import numpy as np
from datasets import load_dataset

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join('data', 'raw')


def load_sova_audiobooks(
    max_samples: int = 50,
) -> list[dict]:
    """
    Load SOVA RuDevices Audiobooks (streaming).

    Human-annotated Russian speech from audiobooks.

    Args:
        max_samples: Max samples to load.

    Returns:
        List of sample dicts.
    """
    logger.info('Loading SOVA audiobooks...')
    os.makedirs(DATA_DIR, exist_ok=True)
    ds = load_dataset(
        'dangrebenkin/sova_rudevices_audiobooks',
        split='test', streaming=True,
        cache_dir=DATA_DIR,
    )
    samples = []
    for item in ds:
        if len(samples) >= max_samples:
            break
        samples.append({
            'audio': item['audio']['array'].astype(
                np.float32,
            ),
            'sample_rate': item['audio']['sampling_rate'],
            'reference': item['transcription'],
            'id': item.get('id', len(samples)),
        })
    logger.info(
        'Loaded %d RU samples from SOVA', len(samples),
    )
    return samples
