#!/usr/bin/env python3
"""Dataset loading utilities."""

import logging
import os

import numpy as np
from datasets import load_dataset

import librosa

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


def load_fleurs_ru(
    max_samples: int = 50,
) -> list[dict]:
    """
    Load Google FLEURS Russian.

    Args:
        max_samples: Max samples to load.

    Returns:
        List of sample dicts.
    """
    logger.info('Loading FLEURS Russian...')
    os.makedirs(DATA_DIR, exist_ok=True)
    ds = load_dataset(
        'google/fleurs', 'ru_ru',
        split='test', streaming=True,
        trust_remote_code=True,
        cache_dir=DATA_DIR,
    )

    samples = []
    for item in ds:
        if len(samples) >= max_samples:
            break
        audio = item['audio']['array'].astype(np.float32)
        sr = item['audio']['sampling_rate']

        if sr != 16000:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=16000,
            )
            sr = 16000

        samples.append({
            'audio': audio,
            'sample_rate': sr,
            'reference': item['transcription'],
            'id': item.get('id', len(samples)),
        })

    logger.info('Loaded %d RU samples', len(samples))
    return samples
