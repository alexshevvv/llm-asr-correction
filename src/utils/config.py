#!/usr/bin/env python3
"""Central experiment configuration."""

from dataclasses import dataclass

import torch


@dataclass
class Config:
    """
    Experiment configuration.

    Attributes:
        device: Inference device (cuda/cpu).
        whisper_model: Whisper model size.
        max_samples: Max samples per dataset.
        llm_temperature: LLM sampling temperature.
    """

    device: str = (
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    whisper_model: str = 'base'
    max_samples: int = 50
    llm_temperature: float = 0.1
