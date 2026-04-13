#!/usr/bin/env python3
"""Dataset loaders grouped by source."""

import os

from src.utils.datasets.fleurs import load_fleurs_ru
from src.utils.datasets.librispeech import load_librispeech

DATA_DIR = os.path.join('data', 'raw')

__all__ = [
    'DATA_DIR',
    'load_fleurs_ru',
    'load_librispeech',
]
