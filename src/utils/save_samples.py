#!/usr/bin/env python3
"""Save audio samples for documentation."""

import logging
import os

import soundfile as sf

logger = logging.getLogger(__name__)

SAMPLES_DIR = os.path.join('data', 'samples')


def save_audio_samples(
    samples: list[dict],
    prefix: str = 'sample',
    count: int = 3,
) -> None:
    """
    Save first N audio samples as WAV files.

    Args:
        samples: List of sample dicts with audio.
        prefix: Filename prefix (e.g. 'en', 'ru').
        count: Number of samples to save.
    """
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    saved = 0
    for sample in samples[:count]:
        fname = f'{prefix}_{saved}.wav'
        path = os.path.join(SAMPLES_DIR, fname)
        if not os.path.exists(path):
            sf.write(path, sample['audio'], sample['sample_rate'])
            logger.info('Saved sample: %s', path)
        saved += 1
