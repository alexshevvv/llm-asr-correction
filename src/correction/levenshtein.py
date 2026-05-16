#!/usr/bin/env python3
"""Character-level Levenshtein distance."""


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute character-level Levenshtein distance.
    """

    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(
                curr[j] + 1,
                prev[j + 1] + 1,
                prev[j] + cost,
            ))
        prev = curr
    return prev[-1]


def normalized_levenshtein(s1: str, s2: str) -> float:
    """
    Normalized Levenshtein distance.
    """

    if not s1 and not s2:
        return 0.0
    dist = levenshtein_distance(s1, s2)
    return dist / max(len(s1), len(s2))
