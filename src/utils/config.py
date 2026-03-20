#!/usr/bin/env python3
"""Central experiment configuration."""

from dataclasses import dataclass

import torch


@dataclass
class Config:
    """
    Experiment configuration.

    Controls ASR models, datasets, and LLM settings.
    All parameters can be overridden at instantiation.

    Attributes:
        device: Inference device (cuda/cpu).
        whisper_model: Whisper model size.
        gigaam_model: GigaAM model variant.
        max_samples: Max samples per dataset.
        llm_model: LLM model name for correction.
        llm_temperature: LLM sampling temperature.
    """

    device: str = (
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    whisper_model: str = 'base'
    gigaam_model: str = 'v2_ctc'
    max_samples: int = 50
    llm_model: str = 'llama-3.1-8b-instant'
    llm_temperature: float = 0.1
