#!/usr/bin/env python3
"""Drawing logic for stacked bar chart."""

import matplotlib.pyplot as plt
import pandas as pd

COLORS = {
    'Improved': '#2ecc71',
    'Degraded': '#e74c3c',
    'Unchanged': '#95a5a6',
}


def draw_stacked(
    group_df: pd.DataFrame,
    group_name: str,
    save_path: str,
) -> None:
    """
    Draw and save one stacked 100% bar chart.

    Args:
        group_df: Subset of analysis_df with column 'X'.
        group_name: Title label for ASR family.
        save_path: File path for the saved PNG.
    """
    counts = group_df.groupby('X').agg({
        'Improved': 'sum',
        'Degraded': 'sum',
        'Unchanged': 'sum',
    }).reset_index()

    totals = (
        counts['Improved']
        + counts['Degraded']
        + counts['Unchanged']
    )
    for cat in COLORS:
        counts[f'{cat} %'] = counts[cat] / totals * 100

    fig, ax = plt.subplots(
        figsize=(max(6.0, len(counts) * 1.5), 5),
    )
    bottom = [0.0] * len(counts)
    for cat in COLORS:
        vals = counts[f'{cat} %'].values
        ax.bar(
            counts['X'], vals,
            bottom=bottom, label=cat,
            color=COLORS[cat], edgecolor='white',
        )
        for i, v in enumerate(vals):
            if v > 5:
                ax.text(
                    i, bottom[i] + v / 2,
                    f'{v:.0f}%',
                    ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='white',
                )
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_title(
        f'Sample Outcomes: {group_name}\n'
        '(share across all LLMs)',
        fontsize=13, fontweight='bold',
    )
    ax.set_ylabel('Share (%)', fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(
        fontsize=10, loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
    )
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
