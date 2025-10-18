"""Utility helpers for evaluating ArcOps-Cyber responses."""

from __future__ import annotations

import re
from typing import Tuple


def _normalise(text: str) -> str:
    tokens = re.sub(r"[`\"']", "", text.strip().lower())
    tokens = re.sub(r"\s+", " ", tokens)
    return tokens


def _split_candidates(text: str) -> list[str]:
    chunks = re.split(r",|;|/|\band\b|\bor\b|\n", text, flags=re.IGNORECASE)
    normalised = []
    for chunk in chunks:
        cleaned = _normalise(chunk)
        if cleaned:
            normalised.append(cleaned)
    return normalised or [_normalise(text)]


def score_answer(gold: str, answer: str) -> Tuple[float, str]:
    """Return a deterministic success score (1.0 or 0.0) plus rationale."""

    if not answer:
        return 0.0, "No answer produced."
    gold_norm = _normalise(gold)
    answer_norm = _normalise(answer)
    if not gold_norm:
        return 1.0, "Gold answer empty; treated as success."

    if gold_norm in answer_norm:
        return 1.0, "Exact string match."

    candidates = _split_candidates(gold)
    if candidates and all(candidate in answer_norm for candidate in candidates):
        return 1.0, f"Matched all expected indicators: {', '.join(candidates)}."

    return 0.0, "Answer did not contain required indicator(s)."
