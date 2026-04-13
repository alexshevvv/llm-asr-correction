#!/usr/bin/env python3
"""LibriSpeech dataset loaders."""

import logging
import os

import numpy as np
from datasets import load_dataset

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join('data', 'raw')


def load_librispeech(
    max_samples: int = 50,
) -> list[dict]:
    """
    Load LibriSpeech test-clean.

    Args:
        max_samples: Max samples to load.

    Returns:
        List of sample dicts.
    """
    logger.info('Loading LibriSpeech test-clean...')
    os.makedirs(DATA_DIR, exist_ok=True)
    ds = load_dataset(
        'librispeech_asr', 'clean',
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
            'reference': item['text'],
            'id': item.get('id', len(samples)),
        })
    logger.info('Loaded %d EN samples', len(samples))
    return samples
