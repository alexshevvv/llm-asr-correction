#!/usr/bin/env python3
"""Sber GigaAM ASR wrapper."""

import logging
import os

import numpy as np

from src.asr.base import BaseASR
from src.utils.audio import save_temp_wav

logger = logging.getLogger(__name__)

MAX_DURATION_SEC = 25

try:
    import gigaam
    GIGAAM_AVAILABLE = True
except ImportError:
    GIGAAM_AVAILABLE = False
    logger.warning(
        'gigaam not installed. '
        'GigaAMASR will not work. '
        'Install: pip install git+'
        'https://github.com/salute-developers/GigaAM.git'
    )


class GigaAMASR(BaseASR):
    """Sber GigaAM ASR transcriber.

    Conformer-based model (240M params)

    Attributes:
        model_name: GigaAM model variant.
    """

    def __init__(
        self,
        model_name: str = 'v2_ctc',
    ) -> None:
        """Initialize and load GigaAM model.

        Args:
            model_name: Model variant
                (v2_ctc/v2_rnnt/v1_ctc/v1_rnnt).

        Raises:
            ImportError: If gigaam package is not installed.
        """
        if not GIGAAM_AVAILABLE:
            raise ImportError(
                'gigaam package is required. '
                'Requires Linux + CUDA + torch>=2.5'
            )
        self.model_name = model_name
        logger.info('Loading GigaAM-%s...', model_name)
        self.model = gigaam.load_model(model_name)
        logger.info('GigaAM loaded')

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        **kwargs,
    ) -> str:
        """Transcribe audio to text.

        Args:
            audio: Audio waveform (float32).
            sample_rate: Audio sample rate in Hz.
            **kwargs: Additional parameters.

        Returns:
            Transcribed text.
        """
        max_samples = MAX_DURATION_SEC * sample_rate
        if len(audio) > max_samples:
            logger.warning(
                'Audio too long (%.1fs), truncating to %ds',
                len(audio) / sample_rate,
                MAX_DURATION_SEC,
            )
            audio = audio[:max_samples]

        wav_path = save_temp_wav(audio, sample_rate)
        try:
            text = self.model.transcribe(wav_path)
        finally:
            os.unlink(wav_path)
        return text.strip()
