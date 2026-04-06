#!/usr/bin/env python3
"""ASR module: Whisper, Wav2Vec2, GigaAM."""

from src.asr.base import BaseASR
from src.asr.wav2vec2_transcribe import Wav2Vec2ASR
from src.asr.whisper_transcribe import WhisperASR

__all__ = [
    'BaseASR',
    'Wav2Vec2ASR',
    'WhisperASR',
]

# GigaAM requires Linux + CUDA, import separately:
# from src.asr.gigaam_transcribe import GigaAMASR
