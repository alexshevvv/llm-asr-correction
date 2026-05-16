#!/usr/bin/env python3
"""Phonetic filter: revert LLM substitutions far from ASR."""

from jiwer import process_words

from src.correction.levenshtein import normalized_levenshtein
from src.evaluation.normalize import normalize_text


def apply_phonetic_filter(
    asr_text: str,
    llm_text: str,
    threshold: float = 0.5,
) -> tuple[str, dict]:
    """
    Filter LLM corrections by closeness to ASR.
    """

    asr_norm = normalize_text(asr_text)
    llm_norm = normalize_text(llm_text)
    asr_words = asr_norm.split()
    llm_words = llm_norm.split()

    if not asr_words or not llm_words:
        return asr_text, _empty_stats()

    output = process_words(asr_norm, llm_norm)
    result_words = []
    stats = _empty_stats()

    for chunk in output.alignments[0]:
        if chunk.type == 'equal':
            for i in range(
                chunk.ref_start_idx, chunk.ref_end_idx,
            ):
                result_words.append(asr_words[i])
        elif chunk.type == 'substitute':
            _handle_subs(
                chunk, asr_words, llm_words,
                threshold, result_words, stats,
            )
        elif chunk.type == 'delete':
            for i in range(
                chunk.ref_start_idx, chunk.ref_end_idx,
            ):
                result_words.append(asr_words[i])
                stats['reverted_dels'] += 1
        elif chunk.type == 'insert':
            for j in range(
                chunk.hyp_start_idx, chunk.hyp_end_idx,
            ):
                result_words.append(llm_words[j])
                stats['kept_ins'] += 1

    return ' '.join(result_words), stats


def _handle_subs(
    chunk, asr_words, llm_words,
    threshold, result_words, stats,
) -> None:
    """Process substitution chunk."""
    for i, j in zip(
        range(chunk.ref_start_idx, chunk.ref_end_idx),
        range(chunk.hyp_start_idx, chunk.hyp_end_idx),
    ):
        dist = normalized_levenshtein(
            asr_words[i], llm_words[j],
        )
        if dist <= threshold:
            result_words.append(llm_words[j])
            stats['accepted_subs'] += 1
        else:
            result_words.append(asr_words[i])
            stats['reverted_subs'] += 1


def _empty_stats() -> dict:
    """Return empty stats dict."""
    return {
        'reverted_subs': 0,
        'accepted_subs': 0,
        'reverted_dels': 0,
        'kept_ins': 0,
    }
