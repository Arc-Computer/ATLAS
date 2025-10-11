from __future__ import annotations

from typing import Any, Dict, Optional

from datasets import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from trainers.runtime_dataset import (
    load_runtime_traces,
    sessions_to_rl_records,
)


def get_runtime_trace_dataset(
    tokenizer: PreTrainedTokenizerBase,
    export_path: str,
    eval_split_ratio: float = 0.1,
    dataset_max_samples: Optional[int] = None,
    shuffle: bool = True,
) -> Dict[str, Any]:
    """Load runtime JSONL exports and prepare train/eval splits for RL."""

    sessions = load_runtime_traces(export_path)
    records = sessions_to_rl_records(sessions)

    if dataset_max_samples is not None:
        records = records[:dataset_max_samples]

    dataset = Dataset.from_list(records)
    if shuffle:
        dataset = dataset.shuffle(seed=42)

    if eval_split_ratio and 0 < eval_split_ratio < 1:
        dataset_dict = dataset.train_test_split(test_size=eval_split_ratio, seed=42)
        return {
            "train_dataset": dataset_dict["train"],
            "eval_dataset": dataset_dict["test"],
        }

    return {"train_dataset": dataset, "eval_dataset": None}
