"""Utilities for building GSM8K datasets for GKD and GRPO experiments."""

from __future__ import annotations

from typing import Dict, Optional

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from trainers.math_gkd_dataset import (
    MathGKDDatasetConfig,
    build_math_gkd_dataset,
)


def build_gsm8k_gkd_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str = "gsm8k",
    subset: str = "main",
    train_split: str = "train",
    eval_split: Optional[str] = None,
    train_ratio: float = 0.9,
    seed: int = 42,
    dataset_max_samples: Optional[int] = None,
) -> Dict[str, Dataset]:
    """Construct train/eval datasets formatted for TRL chat trainers using GSM8K."""

    cfg = MathGKDDatasetConfig(
        dataset_name=dataset_name,
        dataset_config=subset,
        train_split=train_split,
        eval_split=eval_split,
        train_ratio=train_ratio,
        seed=seed,
        limit=dataset_max_samples,
    )
    train_ds, eval_ds = build_math_gkd_dataset(cfg)

    def _format(split: Dataset) -> Dataset:
        def _map(example):
            messages = example["messages"]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return {
                "messages": messages,
                "prompt": prompt,
                "final_answer": example.get("final_answer", ""),
                "source_id": example.get("source_id"),
            }

        return split.map(_map)

    formatted_train = _format(train_ds)
    formatted_eval = _format(eval_ds)

    return {
        "train_dataset": formatted_train,
        "eval_dataset": formatted_eval,
    }
