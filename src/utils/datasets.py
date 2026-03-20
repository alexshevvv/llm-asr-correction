#!/usr/bin/env python3
"""Dataset loaders for ASR experiments."""

import logging

import numpy as np
from datasets import load_dataset

from src.utils.audio import resample_audio

logger = logging.getLogger(__name__)


def load_librispeech(
    max_samples: int = 50,
) -> list[dict]:
    """
    Load LibriSpeech test-clean dataset.

    Uses streaming to avoid downloading the full
    dataset (~28 GB).

    Args:
        max_samples: Maximum number of samples.

    Returns:
        List of dicts with audio, sample_rate,
        reference, and id keys.
    """
    logger.info('Loading LibriSpeech test-clean...')
    ds = load_dataset(
        'librispeech_asr', 'clean',
        split='test', streaming=True,
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

    logger.info(f'Loaded {len(samples)} EN samples')
    return samples


def load_fleurs_ru(
    max_samples: int = 50,
) -> list[dict]:
    """
    Load Google FLEURS Russian dataset.

    Uses streaming to minimize disk usage.

    Args:
        max_samples: Maximum number of samples.

    Returns:
        List of dicts with audio, sample_rate,
        reference, and id keys.
    """
    logger.info('Loading FLEURS Russian...')
    ds = load_dataset(
        'google/fleurs', 'ru_ru',
        split='test', streaming=True,
        trust_remote_code=True,
    )

    samples = []
    for item in ds:
        if len(samples) >= max_samples:
            break
        audio = item['audio']['array'].astype(
            np.float32,
        )
        sr = item['audio']['sampling_rate']

        if sr != 16000:
            audio = resample_audio(audio, sr, 16000)
            sr = 16000

        samples.append({
            'audio': audio,
            'sample_rate': sr,
            'reference': item['transcription'],
            'id': item.get('id', len(samples)),
        })

    logger.info(f'Loaded {len(samples)} RU samples')
    return samples
