#!/usr/bin/env python3
"""LLM client for ASR error correction."""

import logging
import os
import time

from openai import OpenAI

from src.correction.clean_response import clean_response
from src.correction.prompts import get_prompts

logger = logging.getLogger(__name__)


def create_client(api_key: str = '') -> OpenAI:
    """
    Create HuggingFace Inference API client.

    Args:
        api_key: HF token. Falls back to env var.

    Returns:
        Configured OpenAI-compatible client.
    """
    key = api_key or os.getenv('HF_TOKEN', '')
    return OpenAI(
        base_url='https://router.huggingface.co/v1',
        api_key=key,
    )


def correct_with_llm(
    client: OpenAI,
    text: str,
    language: str = 'en',
    model: str = 'Qwen/Qwen2.5-72B-Instruct',
    temperature: float = 0.1,
    max_retries: int = 3,
) -> str:
    """
    Correct ASR transcription using LLM.

    Args:
        client: HF Inference API client.
        text: ASR transcription to correct.
        language: Language code (en or ru).
        model: HuggingFace model identifier.
        temperature: Sampling temperature.
        max_retries: Max retries on rate limit.

    Returns:
        Corrected transcription.
    """
    system_prompt, user_prompt = get_prompts(
        text, language,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                temperature=temperature,
                max_tokens=500,
            )
            result = response.choices[0].message.content
            if result:
                return clean_response(
                    result.strip(), text,
                )
            return text
        except Exception as e:
            if '429' in str(e) or 'rate' in str(e).lower():
                wait = 2 ** attempt * 10
                logger.warning(
                    'Rate limited, waiting %ds...', wait,
                )
                time.sleep(wait)
            else:
                logger.error('LLM error: %s', e)
                return text

    logger.error('Max retries exceeded')
    return text
