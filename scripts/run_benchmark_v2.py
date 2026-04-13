#!/usr/bin/env python3
"""
Registry-driven benchmark: baseline ASR + LLM correction.

End-to-end script wiring together the new architecture:
- Baseline matrix via scripts.benchmark_baselines_v2
- LLM correction via scripts.benchmark_correction_v2
- Analysis DataFrame via src.visualization.analysis

Requires HF_TOKEN in .env. Without it, only baseline runs.
"""

import logging
import os
import sys

from dotenv import load_dotenv

from scripts.benchmark_baselines_v2 import run_baseline_matrix
from scripts.benchmark_correction_v2 import (
    run_correction_matrix,
)
from src.asr.registry_data import ASR_REGISTRY
from src.correction.llm_client import create_client
from src.utils.datasets_registry_data import DATASETS_REGISTRY
from src.visualization.analysis import build_analysis_df

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)

MAX_SAMPLES = 3


def main() -> int:
    """Entry point."""
    baselines = run_baseline_matrix(max_samples=MAX_SAMPLES)
    print(f'\nBaseline matrix: {len(baselines)} experiments')
    for key, df in baselines.items():
        print(f'  {key}: WER={df["wer"].mean():.2%}')

    if not os.getenv('HF_TOKEN'):
        logger.warning(
            'HF_TOKEN not found in env, skipping correction'
        )
        return 0

    client = create_client()
    results = run_correction_matrix(baselines, client)
    analysis_df = build_analysis_df(
        results, ASR_REGISTRY, DATASETS_REGISTRY,
    )
    print(f'\nAnalysis DataFrame: {len(analysis_df)} rows')
    print(analysis_df[[
        'LLM', 'ASR', 'Language',
        'Baseline WER', 'Corrected WER', 'WER Change (%)',
    ]].to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
