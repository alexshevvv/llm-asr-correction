#!/usr/bin/env python3
"""Run ASR baseline experiments."""

import logging

import pandas as pd
from tqdm.auto import tqdm

from src.asr.base import BaseASR
from src.asr.whisper_transcribe import WhisperASR
from src.evaluation import calculate_cer
from src.evaluation import calculate_wer

logger = logging.getLogger(__name__)


def run_baseline(
    asr_model: BaseASR,
    samples: list[dict],
    language: str = 'en',
    model_name: str = 'whisper',
) -> pd.DataFrame:
    """
    Run ASR on samples and compute metrics.

    Args:
        asr_model: ASR model instance.
        samples: List of audio samples.
        language: Language code for Whisper.
        model_name: Label for results table.

    Returns:
        DataFrame with per-sample WER and CER.
    """
    results = []
    desc = f'Baseline {model_name}'

    for sample in tqdm(samples, desc=desc):
        if isinstance(asr_model, WhisperASR):
            hyp = asr_model.transcribe(
                sample['audio'], language=language,
            )
        else:
            hyp = asr_model.transcribe(
                sample['audio'],
                sample_rate=sample['sample_rate'],
            )

        ref = sample['reference']
        results.append({
            'id': sample['id'],
            'reference': ref,
            'hypothesis': hyp,
            'wer': calculate_wer(ref, hyp),
            'cer': calculate_cer(ref, hyp),
            'model': model_name,
            'language': language,
        })

    return pd.DataFrame(results)
