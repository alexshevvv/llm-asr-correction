#!/usr/bin/env python3
"""Dataset loaders grouped by source.

Re-exports loader functions so callers can continue to use
`from src.utils.datasets import load_librispeech` as before.
"""

import os

from src.utils.datasets.fleurs import load_fleurs_en
from src.utils.datasets.fleurs import load_fleurs_ru
from src.utils.datasets.librispeech import load_librispeech
from src.utils.datasets.librispeech import load_librispeech_other
from src.utils.datasets.sova import load_sova_audiobooks

DATA_DIR = os.path.join('data', 'raw')

__all__ = [
    'DATA_DIR',
    'load_fleurs_en',
    'load_fleurs_ru',
    'load_librispeech',
    'load_librispeech_other',
    'load_sova_audiobooks',
]
