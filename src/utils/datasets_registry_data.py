#!/usr/bin/env python3
"""Declarative metadata for datasets used in the benchmark."""

from typing import Any

DATASETS_REGISTRY: dict[str, dict[str, Any]] = {
    'librispeech_test_clean': {
        'loader_path': 'src.utils.datasets.load_librispeech',
        'language': 'en',
        'display_name': 'LibriSpeech test-clean',
        'domain': 'read_speech',
        'style': 'clean',
        'source': 'LibriSpeech',
        'profiles': ['local', 'colab_whisper_w2v2'],
    },
    'librispeech_test_other': {
        'loader_path': (
            'src.utils.datasets.load_librispeech_other'
        ),
        'language': 'en',
        'display_name': 'LibriSpeech test-other',
        'domain': 'read_speech',
        'style': 'noisy',
        'source': 'LibriSpeech',
        'profiles': ['local', 'colab_whisper_w2v2'],
    },
    'fleurs_en': {
        'loader_path': 'src.utils.datasets.load_fleurs_en',
        'language': 'en',
        'display_name': 'FLEURS English',
        'domain': 'read_speech',
        'style': 'clean',
        'source': 'Google FLEURS',
        'profiles': ['local', 'colab_whisper_w2v2'],
    },
    'fleurs_ru': {
        'loader_path': 'src.utils.datasets.load_fleurs_ru',
        'language': 'ru',
        'display_name': 'FLEURS Russian',
        'domain': 'read_speech',
        'style': 'clean',
        'source': 'Google FLEURS',
        'profiles': [
            'local', 'colab_whisper_w2v2', 'colab_gigaam',
        ],
    },
    'sova_audiobooks_ru': {
        'loader_path': (
            'src.utils.datasets.load_sova_audiobooks'
        ),
        'language': 'ru',
        'display_name': 'SOVA audiobooks',
        'domain': 'read_speech',
        'style': 'literary',
        'source': 'SOVA.ai',
        'profiles': [
            'local', 'colab_whisper_w2v2', 'colab_gigaam',
        ],
    },
}
