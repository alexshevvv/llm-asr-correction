#!/usr/bin/env python3
"""Wav2Vec2 XLS-R ASR transcription."""

import logging

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2Processor

from src.asr.base import BaseASR

logger = logging.getLogger(__name__)

MODEL_ID = 'jonatasgrosman/wav2vec2-xls-r-1b-russian'


class Wav2Vec2ASR(BaseASR):
    """Wav2Vec2 XLS-R 1B Russian ASR model."""

    def __init__(self, device: str = 'cpu'):
        """
        Initialize Wav2Vec2 model.

        Args:
            device: Device to run inference on.
        """
        logger.info('Loading Wav2Vec2 XLS-R...')
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(
            MODEL_ID,
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(
            MODEL_ID,
        ).to(device)
        self.model.eval()
        logger.info('Wav2Vec2 loaded on %s', device)

    def transcribe(
        self,
        audio: np.ndarray,
        **kwargs,
    ) -> str:
        """
        Transcribe audio with Wav2Vec2.

        Args:
           audio: Audio waveform (16kHz float32).

        Returns:
           Transcribed text.
        """
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors='pt',
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        pred_ids = torch.argmax(logits, dim=-1)
        text = self.processor.batch_decode(pred_ids)[0]
        return text.strip()
