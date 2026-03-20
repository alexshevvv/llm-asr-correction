#!/usr/bin/env python3
"""OpenAI Whisper ASR wrapper."""

import logging

import numpy as np
import whisper

from src.asr.base import BaseASR

logger = logging.getLogger(__name__)


class WhisperASR(BaseASR):
    """
    OpenAI Whisper ASR transcriber.

    Supports 99 languages via encoder-decoder
    Transformer architecture.

    Attributes:
        model_name: Whisper model size.
        device: Inference device.
    """

    def __init__(
        self,
        model_name: str = 'base',
        device: str = 'cuda',
    ) -> None:
        """
        Initialize and load Whisper model.

        Args:
            model_name: Model size
                (tiny/base/small/medium/large).
            device: Inference device (cuda/cpu).
        """
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
