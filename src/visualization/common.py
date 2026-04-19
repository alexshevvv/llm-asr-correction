#!/usr/bin/env python3
"""Shared utilities for visualization."""

import os

import seaborn as sns

RESULTS_DIR = os.path.join('experiments', 'results')

sns.set_theme(
    style='whitegrid',
    palette='husl',
    font_scale=1.1,
)

ASR_GROUPS = {
    'Whisper base': ['Whisper base'],
    'Whisper medium': ['Whisper medium'],
    'GigaAM family': [
        'GigaAM v2 CTC',
        'GigaAM v2 RNNT',
    ],
    'Wav2Vec2 family': [
        'Wav2Vec2 XLS-R 1B',
        'Wav2Vec2 XLS-R 1B EN',
    ],
}

