#!/usr/bin/env python3
"""
Baseline-only benchmark using registries (validation script).

Demonstrates the new registry-driven architecture:
- ASR models come from src.asr.registry_data
- Datasets come from src.utils.datasets_registry_data
- Analysis DataFrame built via build_analysis_df
"""

import logging
import sys

import pandas as pd

from src.asr.registry_data import ASR_REGISTRY
from src.asr.registry_query import list_models
from src.utils.class_loader import resolve_class
from src.utils.dataset_loader import load_dataset_by_key
from src.utils.datasets_registry_data import DATASETS_REGISTRY
from src.utils.datasets_registry_query import list_datasets
from src.evaluation.metrics import calculate_wer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)

MAX_SAMPLES = 5


def run_baseline_matrix() -> dict[str, pd.DataFrame]:
    """
    Run baseline ASR for every compatible (ASR, dataset) pair.

    Returns:
        Dict '<asr_key>__<dataset_key>' -> baseline DataFrame.
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

            samples = load_dataset_by_key(ds_key, MAX_SAMPLES)
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


def main() -> int:
    """Entry point."""
    results = run_baseline_matrix()
    print(f'\nBaseline matrix: {len(results)} experiments')
    for key, df in results.items():
        print(f'  {key}: WER={df["wer"].mean():.2%}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
