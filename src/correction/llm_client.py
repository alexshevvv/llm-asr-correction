#!/usr/bin/env python3
"""LLM client for ASR error correction."""

import logging

from groq import Groq

from src.correction.prompts import get_prompts

logger = logging.getLogger(__name__)


def create_client(api_key: str) -> Groq:
    """
    Create Groq API client.

    Args:
        api_key: Groq API key.

    Returns:
        Configured Groq client instance.
    """
    return Groq(api_key=api_key)


def correct_with_llm(
    client: Groq,
    text: str,
    language: str = 'en',
    model: str = 'llama-3.1-8b-instant',
    temperature: float = 0.1,
) -> str:
    """Correct ASR transcription using LLM.

    Args:
        client: Groq API client.
        text: ASR transcription to correct.
        language: Language code (en or ru).
        model: LLM model name.
        temperature: Sampling temperature.

    Returns:
        Corrected transcription.
    """
    system_prompt, user_prompt = get_prompts(
        text, language,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            temperature=temperature,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error('LLM correction failed: %s', e)
        return text
