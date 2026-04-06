#!/usr/bin/env python3
"""Audio utilities for ASR pipeline."""

import tempfile

import numpy as np
import soundfile as sf

import librosa


def save_temp_wav(
    audio: np.ndarray,
    sample_rate: int = 16000,
) -> str:
    """
    Save numpy audio array to a temporary WAV file.

    Args:
        audio: Audio waveform as numpy array.
        sample_rate: Audio sample rate in Hz.

    Returns:
        Path to the temporary WAV file.
    """
    tmp = tempfile.NamedTemporaryFile(
        suffix='.wav', delete=False,
    )
    sf.write(tmp.name, audio, sample_rate)
    return tmp.name


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = 16000,
) -> np.ndarray:
    """
    Resample audio to target sample rate.

    Args:
        audio: Audio waveform.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled audio array.
    """
    if orig_sr == target_sr:
        return audio

    return librosa.resample(
        audio, orig_sr=orig_sr, target_sr=target_sr,
    )
