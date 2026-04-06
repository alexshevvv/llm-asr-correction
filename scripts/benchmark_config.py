#!/usr/bin/env python3
"""Benchmark constants and model configuration."""

import os

DATA_DIR = os.path.join('data', 'raw')
RESULTS_DIR = os.path.join('experiments', 'results')
CONFIGS_DIR = os.path.join('experiments', 'configs')
LOGS_DIR = os.path.join('experiments', 'logs')

LLM_MODELS = [
    {
        'id': 'Qwen/Qwen2.5-7B-Instruct',
        'name': 'Qwen2.5 7B',
    },
    {
        'id': 'Qwen/Qwen2.5-72B-Instruct',
        'name': 'Qwen2.5 72B',
    },
    {
        'id': 'openai/gpt-oss-120b',
        'name': 'GPT-OSS 120B',
    },
    {
        'id': 'Qwen/Qwen3-235B-A22B',
        'name': 'Qwen3 235B',
    },
    {
        'id': 'deepseek-ai/DeepSeek-V3-0324',
        'name': 'DeepSeek V3',
    },
]
