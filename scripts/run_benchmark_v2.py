#!/usr/bin/env python3
"""
Registry-driven baseline benchmark (thin orchestration).

Uses scripts.benchmark_baselines_v2.run_baseline_matrix
to exercise the new registry architecture end-to-end.
LLM correction will be wired in a follow-up commit.
"""

import logging
import sys

from scripts.benchmark_baselines_v2 import run_baseline_matrix

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
    return 0


if __name__ == '__main__':
    sys.exit(main())
