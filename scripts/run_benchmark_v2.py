#!/usr/bin/env python3
"""
Registry-driven benchmark: baseline ASR + LLM correction.

"""

import logging
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

from scripts.benchmark_baselines_v2 import run_baseline_matrix
from scripts.benchmark_correction_v2 import (
    run_correction_matrix,
)
from src.asr.registry_data import ASR_REGISTRY
from src.correction.llm_client import create_client
from src.utils.datasets_registry_data import DATASETS_REGISTRY
from src.visualization.analysis import build_analysis_df
from src.visualization import plot_all
from scripts.benchmark_save import save_correction_csvs


load_dotenv()

RESULTS_DIR = os.path.join('experiments', 'results')
LOG_DIR = os.path.join('experiments', 'logs')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_path = os.path.join(LOG_DIR, f'run_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path),
    ],
)
logger = logging.getLogger(__name__)

MAX_SAMPLES = 30


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

    csv_path = os.path.join(
        RESULTS_DIR, f'analysis_{timestamp}.csv',
    )
    analysis_df.to_csv(csv_path, index=False)
    logger.info('Saved analysis CSV: %s', csv_path)

    save_correction_csvs(results, RESULTS_DIR)

    plot_all(analysis_df, baselines)

    print(f'\nAnalysis: {len(analysis_df)} rows')
    print(analysis_df[[
        'LLM', 'ASR', 'Dataset', 'Language',
        'Baseline WER', 'Corrected WER',
        'WER Change (%)',
    ]].to_string(index=False))
    print(f'\nResults saved to {RESULTS_DIR}/')
    print(f'Log saved to {log_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
