#!/usr/bin/env python3
"""Post-processing for LLM correction responses."""

import re

PREFIXES = [
    'Corrected:', 'Corrected text:',
    'Here is', "Here's",
    'The corrected', 'Fixed:',
    'The original text',
    'Исправленный текст:',
    'Исправлено:', 'Вот исправленный',
    'Since the original', 'It looks like',
]

TAIL_PATTERN = re.compile(
    r'\s*\((?:Note|Примечание|Текст не содержит)'
    r'[\s\S]*?\)\s*$',
    re.IGNORECASE,
)

REFUSAL_MARKERS = [
    'already correct',
    'appears correct',
    'appears to be correct',
    'no ASR errors',
    'no obvious ASR errors',
    'does not contain any',
    'не содержит ошибок',
    'исправления не требуются',
    'ошибок не обнаружено',
]


def clean_response(text: str, original: str) -> str:
    """
    Extract corrected text from LLM response.

    """
    if not text:
        return original

    text_lower = text.lower()
    for marker in REFUSAL_MARKERS:
        if marker in text_lower:
            return original

    text = TAIL_PATTERN.sub('', text).strip()

    if len(text) > len(original) * 1.5:
        first_line = text.split('\n')[0].strip()
        if first_line:
            text = first_line

    for prefix in PREFIXES:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
            break

    if (
        len(text) > 2
        and text[0] in '"\u201c'
        and text[-1] in '"\u201d'
    ):
        text = text[1:-1].strip()

    return text if text else original
