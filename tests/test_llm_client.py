#!/usr/bin/env python3
"""Tests for HuggingFace Inference API client."""

from src.correction.llm_client import create_client
from src.correction.llm_client import correct_with_llm


def test_openai_instance():
    """Client is created with HF base URL."""
    client = create_client(api_key='test_key')
    assert client.base_url is not None
    assert 'huggingface' in str(client.base_url)


def test_usage_hf_router():
    """Base URL points to HF router."""
    client = create_client(api_key='test_key')
    url = str(client.base_url)
    assert 'router.huggingface.co' in url


def test_return_on_error():
    """Returns original text when API fails."""
    client = create_client(api_key='invalid_key')
    result = correct_with_llm(
        client,
        'test text',
        language='en',
        model='nonexistent/model',
        max_retries=1,
    )
    assert result == 'test text'


def test_accepts_ru_language():
    """Function accepts ru language parameter."""
    client = create_client(api_key='invalid')
    result = correct_with_llm(
        client,
        'тестовый текст',
        language='ru',
        model='nonexistent/model',
        max_retries=1,
    )
    assert result == 'тестовый текст'
