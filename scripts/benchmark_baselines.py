#!/usr/bin/env python3
"""Run ASR baseline experiments for benchmark."""

import logging

from src.asr.wav2vec2_transcribe import Wav2Vec2ASR
from src.asr.whisper_transcribe import WhisperASR
from src.utils.config import Config
from src.utils.datasets import load_fleurs_ru
from src.utils.datasets import load_librispeech
from src.utils.save_samples import save_audio_samples
from scripts.baseline_cache import load_cached
from scripts.baseline_cache import save_cache
from scripts.benchmark_io import save_csv
from scripts.run_baseline import run_baseline

logger = logging.getLogger(__name__)

CACHE_MAP = {
    'whisper_en': 'baseline_whisper_en.csv',
    'whisper_ru': 'baseline_whisper_ru.csv',
    'w2v2_ru': 'baseline_w2v2_ru.csv',
}


def run_baselines(
    config: Config,
    use_cache: bool = True,
) -> dict:
    """
    Run ASR baselines on all datasets.

    Args:
        config: Experiment configuration.
        use_cache: Use cached baselines if available.

    Returns:
        Dict of baseline DataFrames.
    """
    en = load_librispeech(config.max_samples)
    ru = load_fleurs_ru(config.max_samples)

    save_audio_samples(en, prefix='en')
    save_audio_samples(ru, prefix='ru')

    if use_cache:
        baselines = {}
        for key, fname in CACHE_MAP.items():
            cached = load_cached(fname)
            if cached is not None:
                baselines[key] = cached
            else:
                break
        if len(baselines) == len(CACHE_MAP):
            logger.info('All baselines from cache')
            return baselines

    whisper = WhisperASR(
        model_name=config.whisper_model,
        device=config.device,
    )
    w2v = Wav2Vec2ASR(device=config.device)

    baselines = {
        'whisper_en': run_baseline(
            whisper, en, 'en', 'whisper-base',
        ),
        'whisper_ru': run_baseline(
            whisper, ru, 'ru', 'whisper-base',
        ),
        'w2v2_ru': run_baseline(
            w2v, ru, 'ru', 'wav2vec2-xls-r-1b',
        ),
    }

    for key, fname in CACHE_MAP.items():
        save_cache(baselines[key], fname)
        save_csv(baselines[key], fname)

    return baselines
