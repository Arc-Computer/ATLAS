from datasets import load_dataset
from typing import Dict, Any, Optional
from transformers import PreTrainedTokenizer


def get_bigmath_rl_dataset(
    tokenizer: PreTrainedTokenizer,
    dataset_id_or_path: str = "open-r1/Big-Math-RL-Verified-Processed",
    dataset_split: str = "train",
    dataset_level_filter: str = "level_4_5",
    dataset_max_samples: Optional[int] = 1000,
) -> Dict[str, Any]:

    dataset = load_dataset(dataset_id_or_path, dataset_level_filter, split=dataset_split)

    if dataset_max_samples is not None:
        dataset = dataset.select(range(min(dataset_max_samples, len(dataset))))

    def format_example(example):
        return {
            "prompt": example["prompt"],
            "ground_truth": example["solution"],
        }

    formatted_dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    return {
        "train_dataset": formatted_dataset,
        "eval_dataset": None,
    }
