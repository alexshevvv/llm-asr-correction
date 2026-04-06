#!/usr/bin/env python3
"""Abstract base class for ASR models."""

from abc import ABC
from abc import abstractmethod

import numpy as np


class BaseASR(ABC):
    """Abstract interface for ASR transcribers."""

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        **kwargs,
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio waveform (float32, 16kHz).
            **kwargs: Model-specific parameters.

        Returns:
            Transcribed text string.
        """
