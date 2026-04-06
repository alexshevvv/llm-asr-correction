#!/usr/bin/env python3
"""Post-processing for LLM correction responses."""

PREFIXES = [
    'Corrected:', 'Corrected text:',
    'Here is', "Here's",
    'The corrected', 'Fixed:',
    'Исправленный текст:',
    'Исправлено:', 'Вот исправленный',
]


def clean_response(text: str, original: str) -> str:
    """
    Extract corrected text from LLM response.

    Args:
        text: Raw LLM response.
        original: Original ASR text for fallback.

    Returns:
        Cleaned corrected text.
    """
    if not text:
        return original

    if len(text) > len(original) * 3:
        first_line = text.split('\n')[0].strip()
        if first_line:
            text = first_line

    for prefix in PREFIXES:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
            break

    if (
        len(text) > 2
        and text[0] in '""\u201c'
        and text[-1] in '""\u201d'
    ):
        text = text[1:-1].strip()

    return text if text else original
