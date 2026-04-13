#!/usr/bin/env python3
"""Registry-driven baseline ASR matrix."""

import logging

import pandas as pd

from src.asr.registry_data import ASR_REGISTRY
from src.asr.registry_query import list_models
from src.evaluation.metrics import calculate_wer
from src.utils.class_loader import resolve_class
from src.utils.dataset_loader import load_dataset_by_key
from src.utils.datasets_registry_data import DATASETS_REGISTRY
from src.utils.datasets_registry_query import list_datasets

logger = logging.getLogger(__name__)


def run_baseline_matrix(
    max_samples: int = 3,
) -> dict[str, pd.DataFrame]:
    """
    Run baseline ASR for every compatible (ASR, dataset) pair.

    Args:
        max_samples: Samples per dataset.

    Returns:
        Dict '<asr_key>__<dataset_key>' -> baseline DataFrame
        with columns 'reference', 'hypothesis', 'wer'.
    """

    results = {}
    asr_keys = list_models(profile='local')
    dataset_keys = list_datasets(profile='local')

    logger.info(
        'Matrix: %d ASR x %d datasets',
        len(asr_keys), len(dataset_keys),
    )

    for asr_key in asr_keys:
        asr_meta = ASR_REGISTRY[asr_key]
        cls = resolve_class(asr_meta['class_path'])
        model = cls(**asr_meta['init_kwargs'])
        logger.info('Loaded %s', asr_meta['display_name'])

        for ds_key in dataset_keys:
            ds_meta = DATASETS_REGISTRY[ds_key]
            if ds_meta['language'] not in asr_meta['languages']:
                continue

            samples = load_dataset_by_key(ds_key, max_samples)
            rows = []
            for sample in samples:
                hyp = model.transcribe(
                    sample['audio'],
                    sample_rate=sample['sample_rate'],
                    language=ds_meta['language'],
                )
                wer_val = calculate_wer(sample['reference'], hyp)
                rows.append({
                    'reference': sample['reference'],
                    'hypothesis': hyp,
                    'wer': wer_val,
                })
            df = pd.DataFrame(rows)
            key = f'{asr_key}__{ds_key}'
            results[key] = df
            logger.info(
                '%s: n=%d, mean WER=%.2f%%',
                key, len(df), df['wer'].mean() * 100,
            )

    return results
