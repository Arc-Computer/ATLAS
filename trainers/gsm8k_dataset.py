"""Utilities for building GSM8K datasets for GKD and GRPO experiments."""

from __future__ import annotations

from typing import Callable, Dict, Optional

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from trainers.math_gkd_dataset import (
    MathGKDDatasetConfig,
    build_math_gkd_dataset,
)


def _build_base_dataset(
    dataset_name: str,
    subset: str,
    train_split: str,
    eval_split: Optional[str],
    train_ratio: float,
    seed: int,
    dataset_max_samples: Optional[int],
) -> tuple[Dataset, Dataset]:
    cfg = MathGKDDatasetConfig(
        dataset_name=dataset_name,
        dataset_config=subset,
        train_split=train_split,
        eval_split=eval_split,
        train_ratio=train_ratio,
        seed=seed,
        limit=dataset_max_samples,
    )
    return build_math_gkd_dataset(cfg)


def _format_with_prompt(
    split: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    formatter: Callable[[dict, list[dict]], Dict],
) -> Dataset:
    def _map(example: dict) -> Dict:
        messages = example["messages"]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        row = {
            "prompt": prompt,
            "source_id": example.get("source_id"),
        }
        row.update(formatter(example, messages))
        return row

    return split.map(_map)


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

    train_ds, eval_ds = _build_base_dataset(
        dataset_name,
        subset,
        train_split,
        eval_split,
        train_ratio,
        seed,
        dataset_max_samples,
    )

    def _gkd_formatter(example: dict, messages: list[dict]) -> Dict:
        return {
            "messages": messages,
            "final_answer": example.get("final_answer", ""),
        }

    formatted_train = _format_with_prompt(train_ds, tokenizer, _gkd_formatter)
    formatted_eval = _format_with_prompt(eval_ds, tokenizer, _gkd_formatter)

    return {
        "train_dataset": formatted_train,
        "eval_dataset": formatted_eval,
    }


def build_gsm8k_grpo_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str = "gsm8k",
    subset: str = "main",
    train_split: str = "train",
    eval_split: Optional[str] = None,
    train_ratio: float = 0.9,
    seed: int = 42,
    dataset_max_samples: Optional[int] = None,
) -> Dict[str, Dataset]:
    """Construct GSM8K train/eval datasets formatted for TRL GRPO trainers."""

    train_ds, eval_ds = _build_base_dataset(
        dataset_name,
        subset,
        train_split,
        eval_split,
        train_ratio,
        seed,
        dataset_max_samples,
    )

    def _grpo_formatter(example: dict, messages: list[dict]) -> Dict:
        return {
            "meta_messages": messages,
            "ground_truth": example.get("final_answer", ""),
        }

    formatted_train = _format_with_prompt(train_ds, tokenizer, _grpo_formatter)
    formatted_eval = _format_with_prompt(eval_ds, tokenizer, _grpo_formatter)

    return {
        "train_dataset": formatted_train,
        "eval_dataset": formatted_eval,
    }
