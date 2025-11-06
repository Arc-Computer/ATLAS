"""Dataset helper for running GRPO on MetaMathQA."""

from __future__ import annotations

from typing import Dict, Optional

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from trainers.math_gkd_dataset import (
    MathGKDDatasetConfig,
    build_math_gkd_dataset,
)


def build_meta_math_grpo_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str = "meta-math/MetaMathQA",
    train_split: str = "train",
    eval_split: Optional[str] = None,
    train_ratio: float = 0.9,
    seed: int = 42,
    dataset_max_samples: Optional[int] = None,
) -> Dict[str, Dataset]:
    """Construct GRPO train/eval datasets from MetaMathQA."""

    cfg = MathGKDDatasetConfig(
        dataset_name=dataset_name,
        train_split=train_split,
        eval_split=eval_split,
        train_ratio=train_ratio,
        seed=seed,
        limit=dataset_max_samples,
    )
    train_ds, eval_ds = build_math_gkd_dataset(cfg)

    def _format_for_grpo(split: Dataset) -> Dataset:
        def _map_row(example):
            messages = example["messages"]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return {
                "prompt": prompt,
                "ground_truth": example.get("final_answer", ""),
                "meta_messages": messages,
                "source_id": example.get("source_id"),
            }

        return split.map(_map_row)

    formatted_train = _format_for_grpo(train_ds)
    formatted_eval = _format_for_grpo(eval_ds)

    return {
        "train_dataset": formatted_train,
        "eval_dataset": formatted_eval,
    }
