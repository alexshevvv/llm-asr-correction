#!/usr/bin/env python3
"""Google FLEURS dataset loaders."""

import logging
import os

import librosa
import numpy as np
from datasets import load_dataset

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join('data', 'raw')


def _load_lang(
    config_name: str,
    label: str,
    lang_log: str,
    max_samples: int,
) -> list[dict]:
    """Internal loader for any FLEURS language config."""
    logger.info('Loading FLEURS %s...', label)
    os.makedirs(DATA_DIR, exist_ok=True)
    ds = load_dataset(
        'google/fleurs', config_name,
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
    logger.info(
        'Loaded %d %s samples', len(samples), lang_log,
    )
    return samples


def load_fleurs_ru(max_samples: int = 50) -> list[dict]:
    """Load Google FLEURS Russian (streaming)."""
    return _load_lang(
        config_name='ru_ru',
        label='Russian',
        lang_log='RU',
        max_samples=max_samples,
    )


def load_fleurs_en(max_samples: int = 50) -> list[dict]:
    """Load Google FLEURS English US (streaming)."""
    return _load_lang(
        config_name='en_us',
        label='English',
        lang_log='EN',
        max_samples=max_samples,
    )
