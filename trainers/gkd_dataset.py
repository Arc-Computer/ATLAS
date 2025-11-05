"""Dataset pipeline for GKD training from Atlas runtime traces.

This module provides utilities to stream Atlas sessions from Postgres and convert
them to the multi-turn conversation format expected by TRL's GKDTrainer.

The dataset leverages atlas.training_data for direct database access, eliminating
JSONL export drift and enabling efficient filtering via PostgreSQL indexes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from datasets import Dataset

from trainers.postgres_runtime_dataset import stream_conversations_from_postgres

logger = logging.getLogger(__name__)


def load_gkd_conversations(
    db_url: str,
    *,
    min_reward: float = 0.8,
    learning_key: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    status_filters: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Stream Atlas sessions from Postgres and convert to GKD conversation format.

    This function loads sessions with trajectory_events (multi-turn interactions)
    and converts them to the TRL-compatible format expected by GKDTrainer:

    Message role mapping:
    - Teacher guidance → assistant (demonstrates correct behavior)
    - Student attempts → user (learns from teacher)

    Args:
        db_url: PostgreSQL connection string (e.g., "postgresql://user:pass@host:5432/db")
        min_reward: Minimum session reward threshold for filtering (default: 0.8)
        learning_key: Optional learning key to filter by specific task types
        limit: Maximum number of sessions to load (default: unlimited)
        offset: Number of sessions to skip (for pagination)
        status_filters: List of session statuses to include (default: ["succeeded"])

    Returns:
        List of conversation records in TRL format:
        [{
            "messages": [{"role": "user|assistant", "content": "..."}],
            "session_id": "...",
            "learning_key": "...",
            "reward": 0.95,
            "metadata": {...}
        }]

    Example:
        >>> conversations = load_gkd_conversations(
        ...     db_url="postgresql://localhost:5432/atlas",
        ...     min_reward=0.8,
        ...     learning_key="crm_workflows",
        ...     limit=1000
        ... )
        >>> print(f"Loaded {len(conversations)} conversations")
        Loaded 847 conversations
    """
    if status_filters is None:
        status_filters = ["succeeded"]

    # Use existing postgres_runtime_dataset infrastructure
    # This already handles:
    # - Direct database access via get_training_sessions()
    # - trajectory_events extraction
    # - Message role mapping (teacher→assistant, student→user)
    # - Conversation format conversion
    conversations = stream_conversations_from_postgres(
        db_url,
        min_reward=min_reward,
        limit=limit,
        offset=offset,
        learning_key=learning_key,
        status_filters=status_filters,
        review_status_filters=None,
        include_trajectory_events=True,  # Critical for multi-turn GKD
        include_learning_data=True,  # Include learning metrics for analysis
    )

    if not conversations:
        logger.warning(
            "No conversations found for min_reward=%.2f, learning_key=%s",
            min_reward,
            learning_key,
        )
    else:
        logger.info(
            "Loaded %d conversations for GKD training (min_reward=%.2f)",
            len(conversations),
            min_reward,
        )

    return conversations


def build_gkd_dataset(
    db_url: str,
    *,
    min_reward: float = 0.8,
    learning_key: Optional[str] = None,
    limit: Optional[int] = None,
    eval_split: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """
    Build train/eval HuggingFace datasets for GKD training from Postgres.

    This function:
    1. Streams sessions from Postgres via get_training_sessions()
    2. Filters by reward threshold and learning key
    3. Converts to TRL-compatible conversation format
    4. Splits into train/eval sets
    5. Returns HuggingFace Dataset objects ready for GKDTrainer

    Args:
        db_url: PostgreSQL connection string
        min_reward: Minimum reward threshold (default: 0.8 for high-quality traces)
        learning_key: Optional task-specific filter (e.g., "crm_workflows")
        limit: Maximum total conversations to load (None = unlimited)
        eval_split: Fraction of data for evaluation (default: 0.1 = 10%)
        shuffle: Whether to shuffle dataset before splitting (default: True)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (train_dataset, eval_dataset) as HuggingFace Dataset objects

    Raises:
        ValueError: If no conversations are available for the specified filters

    Example:
        >>> train_ds, eval_ds = build_gkd_dataset(
        ...     db_url="postgresql://localhost:5432/atlas",
        ...     min_reward=0.8,
        ...     eval_split=0.15,
        ... )
        >>> print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")
        Train: 850, Eval: 150
        >>> print(train_ds[0]["messages"][:2])
        [
            {'role': 'system', 'content': 'Task: Create contact for John Smith\\nPlan:\\n- 1: Search contacts...'},
            {'role': 'assistant', 'content': 'First, let me guide you through...'}
        ]
    """
    conversations = load_gkd_conversations(
        db_url=db_url,
        min_reward=min_reward,
        learning_key=learning_key,
        limit=limit,
    )

    if not conversations:
        raise ValueError(
            f"No conversations available for min_reward={min_reward}, "
            f"learning_key={learning_key}. Try lowering min_reward or checking "
            f"database connectivity."
        )

    # Convert to HuggingFace Dataset
    full_dataset = Dataset.from_list(conversations)

    if shuffle:
        full_dataset = full_dataset.shuffle(seed=seed)

    # Split train/eval
    if eval_split and 0 < eval_split < 1:
        split_dataset = full_dataset.train_test_split(
            test_size=eval_split,
            seed=seed,
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    else:
        # No eval split requested
        train_dataset = full_dataset
        eval_dataset = Dataset.from_list([])  # Empty eval set

    logger.info(
        "Built GKD datasets: train=%d, eval=%d (split=%.1f%%)",
        len(train_dataset),
        len(eval_dataset),
        eval_split * 100,
    )

    return train_dataset, eval_dataset


# For backward compatibility and explicit hydra instantiation
def get_gkd_postgres_dataset(
    db_url: str,
    *,
    min_reward: float = 0.8,
    learning_key: Optional[str] = None,
    limit: Optional[int] = None,
    eval_split: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
) -> Dict[str, Dataset]:
    """
    Hydra-compatible entry point for GKD dataset instantiation.

    Returns a dict with 'train_dataset' and 'eval_dataset' keys, matching
    the format expected by train.py's Hydra configuration system.

    Args:
        db_url: PostgreSQL connection string
        min_reward: Minimum reward threshold (default: 0.8)
        learning_key: Optional task-specific filter
        limit: Maximum conversations to load
        eval_split: Evaluation split ratio (default: 0.1)
        shuffle: Shuffle before splitting (default: True)
        seed: Random seed (default: 42)

    Returns:
        Dict with keys:
        - 'train_dataset': HuggingFace Dataset for training
        - 'eval_dataset': HuggingFace Dataset for evaluation

    Example (in configs/data/gkd_postgres.yaml):
        ```yaml
        _target_: trainers.gkd_dataset.get_gkd_postgres_dataset
        db_url: ${oc.env:ATLAS_DB_URL}
        min_reward: 0.8
        eval_split: 0.1
        ```
    """
    train_ds, eval_ds = build_gkd_dataset(
        db_url=db_url,
        min_reward=min_reward,
        learning_key=learning_key,
        limit=limit,
        eval_split=eval_split,
        shuffle=shuffle,
        seed=seed,
    )

    return {
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
    }
