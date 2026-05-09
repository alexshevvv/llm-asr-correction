#!/usr/bin/env python3
"""LLM effect on error types: delta chart."""

import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.evaluation.error_classification import (
    classify_errors,
)

logger = logging.getLogger(__name__)

ETYPE_COLORS = ['#e74c3c', '#3498db', '#f39c12']


def draw_error_delta(
    results: dict[str, pd.DataFrame],
    results_dir: str,
) -> None:
    """Plot delta S/I/D (baseline - corrected) by LLM."""
    rows = []
    for key, df in results.items():
        if len(df) == 0:
            continue
        parts = key.split('__')
        if len(parts) != 3:
            continue
        for _, row in df.iterrows():
            bl = classify_errors(
                row['reference'], row['hypothesis'],
            )
            cr = classify_errors(
                row['reference'], row['corrected'],
            )
            rows.append({
                'LLM': parts[0],
                'Substitutions': (
                    bl['substitutions']
                    - cr['substitutions']
                ),
                'Insertions': (
                    bl['insertions']
                    - cr['insertions']
                ),
                'Deletions': (
                    bl['deletions']
                    - cr['deletions']
                ),
            })
    delta = (
        pd.DataFrame(rows)
        .groupby('LLM')[
            ['Substitutions', 'Insertions', 'Deletions']
        ]
        .mean()
        .reset_index()
    )
    melt = delta.melt(
        id_vars='LLM',
        var_name='Error Type',
        value_name='Mean Delta',
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=melt, x='LLM', y='Mean Delta',
        hue='Error Type', ax=ax,
        palette=ETYPE_COLORS, edgecolor='white',
    )
    ax.axhline(y=0, color='black', ls='--', lw=1)
    ax.set_title(
        'LLM Effect on Error Types\n'
        '(positive = reduced, negative = added)',
        fontsize=13, fontweight='bold',
    )
    ax.set_xlabel('')
    ax.legend(title='Error Type', fontsize=9)
    plt.xticks(rotation=15)
    plt.tight_layout()
    path = os.path.join(
        results_dir, 'plot_error_types_delta.png',
    )
    fig.savefig(path, dpi=150)
    logger.info('Saved: %s', path)
    plt.close(fig)
