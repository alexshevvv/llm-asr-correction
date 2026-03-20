#!/usr/bin/env python3
"""ASR quality metrics: WER and CER."""

from jiwer import cer as _cer
from jiwer import wer as _wer

from src.evaluation.normalize import normalize_text


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate.

    WER = (S + D + I) / N, where S = substitutions,
    D = deletions, I = insertions, N = reference words.

    Args:
        reference: Ground truth transcription.
        hypothesis: ASR output transcription.

    Returns:
        WER value as float (0.0 to 1.0+).
    """
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    if not ref_norm:
        return 1.0 if hyp_norm else 0.0

    return _wer(ref_norm, hyp_norm)


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate.

    Args:
        reference: Ground truth transcription.
        hypothesis: ASR output transcription.

    Returns:
        CER value as float (0.0 to 1.0+).
    """
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    if not ref_norm:
        return 1.0 if hyp_norm else 0.0

    return _cer(ref_norm, hyp_norm)
