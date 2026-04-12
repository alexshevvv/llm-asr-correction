#!/usr/bin/env python3
"""Declarative metadata for LLM correction models."""

from typing import Any

LLM_REGISTRY: dict[str, dict[str, Any]] = {
    'qwen2_5_7b': {
        'hf_id': 'Qwen/Qwen2.5-7B-Instruct',
        'display_name': 'Qwen2.5 7B',
        'family': 'Alibaba',
        'params': '7B',
        'architecture': 'dense',
    },
    'qwen2_5_72b': {
        'hf_id': 'Qwen/Qwen2.5-72B-Instruct',
        'display_name': 'Qwen2.5 72B',
        'family': 'Alibaba',
        'params': '72B',
        'architecture': 'dense',
    },
    'qwen3_235b': {
        'hf_id': 'Qwen/Qwen3-235B-A22B',
        'display_name': 'Qwen3 235B',
        'family': 'Alibaba',
        'params': '235B',
        'architecture': 'MoE',
    },
    'deepseek_v3': {
        'hf_id': 'deepseek-ai/DeepSeek-V3-0324',
        'display_name': 'DeepSeek V3',
        'family': 'DeepSeek',
        'params': '685B',
        'architecture': 'MoE',
    },
    'gpt_oss_120b': {
        'hf_id': 'openai/gpt-oss-120b',
        'display_name': 'GPT-OSS 120B',
        'family': 'OpenAI',
        'params': '120B',
        'architecture': 'MoE',
    },
    'llama_3_3_70b': {
        'hf_id': 'meta-llama/Llama-3.3-70B-Instruct',
        'display_name': 'Llama 3.3 70B',
        'family': 'Meta',
        'params': '70B',
        'architecture': 'dense',
    },
}
