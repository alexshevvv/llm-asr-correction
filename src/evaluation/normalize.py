#!/usr/bin/env python3
"""Text normalization for ASR evaluation metrics"""

import re


def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
) -> str:
    """
    Normalize text for WER/CER calculation.

    Args:
        text: Input text to normalize.
        lowercase: Convert to lowercase.
        remove_punctuation: Remove punctuation marks.

    Returns:
        Normalized text string with single spaces.
    """
    if lowercase:
        text = text.lower()

    if remove_punctuation:
        text = re.sub(
            r'[.,!?;:"\'()\[\]{}\-—–/]',
            '',
            text,
        )

    text = re.sub(r'\s+', ' ', text).strip()
    return text
