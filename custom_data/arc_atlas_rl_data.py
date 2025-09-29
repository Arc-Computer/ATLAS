from datasets import load_dataset
from typing import Dict, Any, Optional
from transformers import PreTrainedTokenizer


def get_arc_atlas_rl_dataset(
    tokenizer: PreTrainedTokenizer,
    dataset_id_or_path: str = "Arc-Intelligence/Arc-ATLAS-Teach-v1",
    dataset_split: str = "rl",
    dataset_max_samples: Optional[int] = None,
    eval_split_ratio: float = 0.1,
) -> Dict[str, Any]:
    
    dataset = load_dataset(dataset_id_or_path, data_files="curriculum/arc_atlas_teach_rl.jsonl", split="train")
    
    if dataset_max_samples is not None:
        dataset = dataset.select(range(min(dataset_max_samples, len(dataset))))
    
    def format_example(example):
        prompt = (
            example.get("prompt")
            or example.get("problem_text")
            or example.get("question")
            or example.get("input")
            or ""
        )
        ground_truth = (
            example.get("ground_truth")
            or example.get("answer")
            or example.get("solution")
            or ""
        )
        additional_context = {
            k: example[k]
            for k in ["student_approach", "teacher_diagnosis", "teacher_teaching"]
            if k in example
        }
        return {
            "prompt": prompt,
            "ground_truth": ground_truth,
            "additional_context": additional_context,
        }
    
    formatted_dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    
    dataset_dict = formatted_dataset.train_test_split(test_size=eval_split_ratio, seed=42)
    
    return {
        "train_dataset": dataset_dict["train"],
        "eval_dataset": dataset_dict["test"],
    }