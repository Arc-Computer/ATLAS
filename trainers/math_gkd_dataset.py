"""Utilities for building math reasoning datasets for Atlas GKD validation.

This module prepares public math datasets (MetaMathQA by default) in the
chat-style format expected by TRL's ``GKDTrainer``. Loading does not require
Atlas' Postgres traces, enabling lightweight validation runs before moving
to production data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset

SYSTEM_PROMPT = (
    "You are a math reasoning expert. Work step-by-step and present a final "
    "answer in the form 'Final Answer: <value>'."
)


@dataclass(frozen=True)
class MathGKDDatasetConfig:
    """Configuration for building the math validation dataset."""

    dataset_name: str = "meta-math/MetaMathQA"
    train_split: str = "train"
    eval_split: Optional[str] = None
    train_ratio: float = 0.9
    seed: int = 42
    limit: Optional[int] = None


def build_math_gkd_dataset(
    config: MathGKDDatasetConfig | None = None,
) -> Tuple[Dataset, Dataset]:
    """
    Create train/eval datasets formatted for ``GKDTrainer`` using MetaMathQA.

    The resulting datasets contain:
      - ``messages``: list of chat turns (system, user, assistant)
      - ``final_answer``: canonical numeric/string answer for metric checks
      - ``source_id``: original dataset identifier for traceability

    Args:
        config: Optional overrides for the dataset source and split behaviour.

    Returns:
        ``(train_dataset, eval_dataset)`` as ``datasets.Dataset`` instances.
    """
    cfg = config or MathGKDDatasetConfig()
    raw = _load_base_dataset(cfg)
    filtered = _filter_missing_fields(raw)
    prepared = filtered.map(_format_example, remove_columns=filtered.column_names)

    if cfg.limit:
        prepared = prepared.select(range(min(cfg.limit, len(prepared))))

    if cfg.eval_split:
        train_ds = prepared["train"]
        eval_ds = prepared["eval"]
    else:
        split = prepared.train_test_split(
            test_size=1.0 - cfg.train_ratio,
            shuffle=True,
            seed=cfg.seed,
        )
        train_ds, eval_ds = split["train"], split["test"]

    return train_ds, eval_ds


def normalize_math_answer(answer: str) -> str:
    """Public helper to normalize answers for downstream metric computation."""
    return _normalize_answer(answer)


def extract_answer_from_text(text: str) -> str:
    """Best-effort extraction of a final answer string from generated text."""
    if not text:
        return ""
    candidate = _parse_answer_from_solution(text)
    if candidate:
        return _normalize_answer(candidate)
    tail = text.strip().splitlines()[-1]
    return _normalize_answer(tail)


def _load_base_dataset(cfg: MathGKDDatasetConfig) -> Dataset | DatasetDict:
    """Load and optionally combine train/eval splits from Hugging Face hub."""
    if cfg.eval_split:
        dataset = load_dataset(
            cfg.dataset_name,
            split={
                "train": cfg.train_split,
                "eval": cfg.eval_split,
            },
        )
    else:
        dataset = load_dataset(cfg.dataset_name, split=cfg.train_split)

    return dataset


def _format_example(example: dict) -> dict:
    """Convert raw MetaMathQA entry into chat messages with metadata."""
    question = _get_question_text(example)
    solution = _get_solution_text(example)
    final_answer = _extract_final_answer(
        example.get("final_answer"),
        example.get("answer"),
        solution,
    )

    user_content = question.strip()
    if not user_content:
        raise ValueError("MetaMathQA record missing question/problem field.")

    assistant_content = solution.strip()
    if not assistant_content:
        raise ValueError("MetaMathQA record missing solution rationale.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    return {
        "messages": messages,
        "final_answer": final_answer,
        "source_id": example.get("id") or example.get("question_id")
        or example.get("problem_id"),
    }


def _extract_final_answer(
    explicit_answer: Optional[str],
    backup_answer: Optional[str],
    solution: str,
) -> str:
    """Derive a canonical final answer for accuracy evaluation."""
    for candidate in _iterate_candidates(
        explicit_answer,
        backup_answer,
        _parse_answer_from_solution(solution),
    ):
        normalized = _normalize_answer(candidate)
        if normalized:
            return normalized
    return ""


def _iterate_candidates(*candidates: Optional[str]) -> Iterable[str]:
    for candidate in candidates:
        if candidate:
            yield candidate


def _get_question_text(example: dict) -> str:
    """Extract question text across known MetaMathQA schema variants."""
    for key in ("question", "problem", "prompt", "query", "original_question"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _get_solution_text(example: dict) -> str:
    """Extract solution/rationale text across known MetaMathQA schema variants."""
    for key in ("solution", "cot_solution", "rationale", "answer", "response"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


_FINAL_ANSWER_PATTERN = re.compile(
    r"(Final Answer|####)\s*[:=]?\s*(?P<answer>[-+–]?[\d\w()./ ^]+)",
    flags=re.IGNORECASE,
)


def _parse_answer_from_solution(solution: str) -> Optional[str]:
    """Attempt to parse a trailing 'Final Answer' or '#### value' statement."""
    if not solution:
        return None
    match = _FINAL_ANSWER_PATTERN.search(solution)
    if match:
        return match.group("answer").strip()
    # fallback to last line heuristic
    tail = solution.strip().splitlines()[-1]
    return tail.rsplit(":", 1)[-1].strip() if tail else None


def _normalize_answer(answer: str) -> str:
    """Normalize answers for easier equality checks."""
    cleaned = re.sub(r"\s+", " ", answer.strip())
    # Remove trailing punctuation commonly added in solutions
    cleaned = cleaned.rstrip(".")
    return cleaned


def _filter_missing_fields(dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
    """Remove entries without question/problem or solution fields."""

    def _has_required_fields(example: dict) -> bool:
        question = _get_question_text(example)
        solution = _get_solution_text(example)
        return bool(question and solution)

    if isinstance(dataset, DatasetDict):
        return dataset.filter(_has_required_fields)
    return dataset.filter(_has_required_fields)
