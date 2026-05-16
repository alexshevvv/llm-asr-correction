#!/usr/bin/env python3
"""OpenAI Whisper ASR wrapper."""

import logging

import numpy as np
import whisper

import torch

from src.asr.base import BaseASR

logger = logging.getLogger(__name__)


class WhisperASR(BaseASR):
    """
    OpenAI Whisper ASR transcriber.

    Attributes:
        model_name: Whisper model size.
        device: Inference device.
    """

    def __init__(
        self,
        model_name: str = 'base',
        device: str | None = None,
    ) -> None:
        """
        Initialize and load Whisper model.

        Args:
            model_name: Model size
                (tiny/base/small/medium/large).
            device: Inference device (cuda/cpu).
                If None, autodetect based on torch.
        """

        if device is None:
            device = (
                'cuda'
                if torch.cuda.is_available()
                else 'cpu'
            )
        self.model_name = model_name
        self.device = device
        logger.info(
            'Loading Whisper-%s on %s',
            model_name, device,
        )
        self.model = whisper.load_model(
            model_name, device=device,
        )
        logger.info('Whisper loaded')

    def transcribe(
        self,
        audio: np.ndarray,
        language: str = 'en',
        **kwargs,
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio waveform (float32).
            language: Language code (en, ru, etc.).
            **kwargs: Additional Whisper parameters.

        Returns:
            Transcribed text.
        """
        result = self.model.transcribe(
            audio,
            language=language,
            fp16=(self.device == 'cuda'),
        )
        return result['text'].strip()

    def transcribe_with_confidence(
        self,
        audio: np.ndarray,
        language: str = 'en',
        **kwargs,
    ) -> tuple[str, float]:
        """
        Transcribe audio and return confidence.

        """
        result = self.model.transcribe(
            audio,
            language=language,
            fp16=(self.device == 'cuda'),
        )
        text = result['text'].strip()
        segments = result.get('segments', [])
        if segments:
            avg_logprob = np.mean([
                s['avg_logprob'] for s in segments
            ])
        else:
            avg_logprob = -1.0
        return text, avg_logprob
