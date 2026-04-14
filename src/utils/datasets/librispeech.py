#!/usr/bin/env python3
"""LibriSpeech dataset loaders."""

import logging
import os

import numpy as np
from datasets import load_dataset

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join('data', 'raw')


def _load_split(
    config_name: str,
    label: str,
    max_samples: int,
) -> list[dict]:
    """Internal loader for any LibriSpeech test split."""
    logger.info('Loading LibriSpeech %s...', label)
    os.makedirs(DATA_DIR, exist_ok=True)
    ds = load_dataset(
        'librispeech_asr', config_name,
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
    logger.info(
        'Loaded %d EN samples from %s',
        len(samples), label,
    )
    return samples


def load_librispeech(max_samples: int = 50) -> list[dict]:
    """Load LibriSpeech test-clean (streaming)."""
    return _load_split(
        config_name='clean',
        label='test-clean',
        max_samples=max_samples,
    )


def load_librispeech_other(
    max_samples: int = 50,
) -> list[dict]:
    """Load LibriSpeech test-other (streaming, noisy)."""
    return _load_split(
        config_name='other',
        label='test-other',
        max_samples=max_samples,
    )
