#!/usr/bin/env python3
"""LLM prompt templates for ASR error correction."""

SYSTEM_PROMPT_EN = (
    'You are an expert at correcting automatic speech '
    'recognition (ASR) errors in English text.\n\n'
    'Rules:\n'
    '1. Fix only clear transcription errors\n'
    '2. Do NOT change meaning or style\n'
    '3. Do NOT add or remove punctuation\n'
    '4. Return ONLY the corrected text, nothing else'
)

SYSTEM_PROMPT_RU = (
    'Ты эксперт по исправлению ошибок автоматического '
    'распознавания речи (ASR) в русском тексте.\n\n'
    'Правила:\n'
    '1. Исправляй только явные ошибки транскрипции\n'
    '2. НЕ меняй смысл и стиль текста\n'
    '3. НЕ добавляй и не удаляй знаки препинания\n'
    '4. Верни ТОЛЬКО исправленный текст, ничего больше'
)

USER_TEMPLATE_EN = (
    'Fix ASR errors:\n\n{text}\n\nCorrected:'
)

USER_TEMPLATE_RU = (
    'Исправь ошибки ASR:\n\n{text}\n\nИсправленный текст:'
)


def get_prompts(
    text: str,
    language: str = 'en',
) -> tuple[str, str]:
    """
    Get system and user prompts for a language.

    Args:
        text: ASR transcription to correct.
        language: Language code (en or ru).

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    if language == 'ru':
        return (
            SYSTEM_PROMPT_RU,
            USER_TEMPLATE_RU.format(text=text),
        )
    return (
        SYSTEM_PROMPT_EN,
        USER_TEMPLATE_EN.format(text=text),
    )
