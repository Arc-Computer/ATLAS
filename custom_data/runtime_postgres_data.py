"""Hydra-friendly wrapper for loading Postgres-backed Atlas conversation datasets."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from trainers.postgres_runtime_dataset import build_conversation_dataset


def get_postgres_runtime_dataset(
    tokenizer: PreTrainedTokenizerBase,
    db_url: str,
    *,
    min_reward: float = 0.0,
    limit: Optional[int] = None,
    offset: int = 0,
    learning_key: Optional[str] = None,
    status_filters: Optional[Sequence[str]] = None,
    review_status_filters: Optional[Sequence[str]] = None,
    eval_split_ratio: float = 0.1,
    shuffle: bool = True,
    dataset_max_sessions: Optional[int] = None,
    include_trajectory_events: bool = True,
    include_learning_data: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """Hydra entry point for the Postgres-backed dataset helper."""

    _ = tokenizer  # Supplied for parity with Hydra signature, unused for now.

    return build_conversation_dataset(
        db_url,
        min_reward=min_reward,
        limit=limit,
        offset=offset,
        learning_key=learning_key,
        status_filters=status_filters,
        review_status_filters=review_status_filters,
        eval_split_ratio=eval_split_ratio,
        shuffle=shuffle,
        dataset_max_sessions=dataset_max_sessions,
        include_trajectory_events=include_trajectory_events,
        include_learning_data=include_learning_data,
        seed=seed,
    )

