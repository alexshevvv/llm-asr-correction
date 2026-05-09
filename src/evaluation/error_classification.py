#!/usr/bin/env python3
"""Word-level error classification: S/I/D."""

from jiwer import process_words

from src.evaluation.normalize import normalize_text


def classify_errors(
    reference: str, hypothesis: str,
) -> dict:
    """
    Classify word-level errors into S/I/D.

    """

    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    if not ref_norm:
        words = hyp_norm.split() if hyp_norm else []
        return {
            'substitutions': 0,
            'insertions': len(words),
            'deletions': 0,
            'hits': 0,
            'ref_words': 0,
        }

    output = process_words(ref_norm, hyp_norm)
    return {
        'substitutions': output.substitutions,
        'insertions': output.insertions,
        'deletions': output.deletions,
        'hits': output.hits,
        'ref_words': (
            output.substitutions
            + output.deletions
            + output.hits
        ),
    }
