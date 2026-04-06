#!/usr/bin/env python3
"""Main benchmark entry point."""

import logging
import os
from datetime import datetime

from dotenv import load_dotenv

from scripts.benchmark_config import RESULTS_DIR
from scripts.benchmark_io import save_csv
from scripts.benchmark_io import save_run_config
from scripts.benchmark_pipeline import run_all_corrections
from scripts.benchmark_baselines import run_baselines
from scripts.benchmark_summary import build_summary
from src.utils.config import Config
from src.visualization import plot_all

load_dotenv()

LOG_DIR = os.path.join('experiments', 'logs')
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


def main() -> None:
    """Run full benchmark pipeline."""
    config = Config()
    logger.info('Config: %s', config)

    # ASR baselines
    baselines = run_baselines(config)
    baselines_info = {
        key: {
            'samples': len(df),
            'mean_wer': float(df['wer'].mean()),
            'errors': int((df['wer'] > 0).sum()),
        }
        for key, df in baselines.items()
    }

    # Save run config
    save_run_config(config, baselines_info)

    for key, info in baselines_info.items():
        logger.info(
            '%s - WER: %.2f%%, errors: %d/%d',
            key, info['mean_wer'] * 100,
            info['errors'], info['samples'],
        )

    # LLM correction
    results = run_all_corrections(baselines)

    # Summary
    summary = build_summary(results)
    save_csv(summary, 'benchmark_summary.csv')

    # Visualizations
    plot_all(results, baselines)

    print('\n' + '=' * 60)
    print('  BENCHMARK COMPLETE')
    print('=' * 60)
    print(summary.to_string(index=False))
    print(f'\nResults: {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
