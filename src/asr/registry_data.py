#!/usr/bin/env python3
"""Declarative metadata for ASR models in the benchmark."""

from typing import Any

ASR_REGISTRY: dict[str, dict[str, Any]] = {
    'whisper_base': {
        'class_path': 'src.asr.whisper_transcribe.WhisperASR',
        'init_kwargs': {'model_name': 'base'},
        'languages': ['en', 'ru'],
        'display_name': 'Whisper base',
        'family': 'OpenAI',
        'params': '74M',
        'architecture': 'encoder-decoder',
        'ram_gb_estimate': 2,
        'profiles': ['local', 'colab_whisper_w2v2'],
    },
    'wav2vec2_xlsr_1b_ru': {
        'class_path': (
            'src.asr.wav2vec2_transcribe.Wav2Vec2ASR'
        ),
        'init_kwargs': {},
        'languages': ['ru'],
        'display_name': 'Wav2Vec2 XLS-R 1B',
        'family': 'Meta / jonatasgrosman',
        'params': '1B',
        'architecture': 'Wav2Vec2-CTC',
        'ram_gb_estimate': 5,
        'profiles': ['local', 'colab_whisper_w2v2'],
    },
}
