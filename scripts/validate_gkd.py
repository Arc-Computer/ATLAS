"""Entry point for validating Atlas GKD training on public math datasets.

This script instantiates the AtlasGKDTrainer with optional pre-loaded datasets
to verify end-to-end configuration before running large-scale training on the
customer CRM benchmark. It follows the recommendations in
``docs/training/offline/gkd-training.mdx`` and the upstream TRL documentation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
import sys
import types
from transformers.utils import import_utils

import_utils.is_apex_available = lambda: False

if "apex" not in sys.modules:
    apex_stub = types.ModuleType("apex")
    apex_stub.amp = None
    sys.modules["apex"] = apex_stub
from datasets import Dataset
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ.setdefault("TRANSFORMERS_NO_APEX", "1")

from trl import GKDConfig

from trainers.gkd_trainer import AtlasGKDTrainer
from trainers.math_gkd_dataset import (
    MathGKDDatasetConfig,
    build_math_gkd_dataset,
    extract_answer_from_text,
    normalize_math_answer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Atlas GKD trainer on MetaMathQA.")
    parser.add_argument(
        "--student",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Student model identifier or local path.",
    )
    parser.add_argument(
        "--teacher",
        default="Qwen/Qwen2.5-14B-Instruct",
        help="Teacher model identifier or local path.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/gkd_math_validation",
        help="Directory to store trainer artifacts.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=1024,
        help="Optional limit on training examples for quick iteration.",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=128,
        help="Optional limit on evaluation examples.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=250,
        help="Maximum training steps for validation run.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=2,
        help="Per-device batch size for training.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation to approximate larger batches.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for the validation run.",
    )
    parser.add_argument(
        "--lmbda",
        type=float,
        default=1.0,
        help="On-policy fraction per GKDConfig.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="KL interpolation coefficient per GKDConfig.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature for student rollouts during training.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens for student generation in evaluation.",
    )
    parser.add_argument(
        "--eval-sample-size",
        type=int,
        default=64,
        help="Subset of evaluation set for accuracy and KL checks.",
    )
    parser.add_argument(
        "--no-baseline-eval",
        action="store_true",
        help="Skip pre/post evaluation to save time.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Force bfloat16 inference when supported.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16 if device == "cuda" else torch.float32

    dataset_cfg = MathGKDDatasetConfig(limit=args.train_limit + args.eval_limit)
    train_dataset, eval_dataset = build_math_gkd_dataset(dataset_cfg)

    if args.train_limit:
        train_dataset = train_dataset.select(range(min(args.train_limit, len(train_dataset))))
    if args.eval_limit:
        eval_dataset = eval_dataset.select(range(min(args.eval_limit, len(eval_dataset))))

    print(f"Loaded datasets: train={len(train_dataset)} eval={len(eval_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load student and teacher models
    model = AutoModelForCausalLM.from_pretrained(
        args.student,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    eval_subset = eval_dataset.select(range(min(args.eval_sample_size, len(eval_dataset))))

    metrics_summary: Dict[str, Dict] = {
        "dataset": {
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset),
            "eval_subset": len(eval_subset),
        },
        "config": {
            "student": args.student,
            "teacher": args.teacher,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "lmbda": args.lmbda,
            "beta": args.beta,
            "temperature": args.temperature,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_new_tokens": args.max_new_tokens,
        },
    }

    if not args.no_baseline_eval:
        print("Running baseline evaluation...")
        metrics_summary["baseline"] = evaluate_model(
            model=model,
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            dataset=eval_subset,
            max_new_tokens=args.max_new_tokens,
        )
        metrics_summary["teacher"] = evaluate_model(
            model=teacher_model,
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            dataset=eval_subset,
            max_new_tokens=args.max_new_tokens,
        )

    gkd_args = GKDConfig(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="no",
        logging_steps=10,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lmbda=args.lmbda,
        beta=args.beta,
        temperature=args.temperature,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        bf16=args.bf16,
        fp16=(not args.bf16 and device == "cuda"),
        do_predict=False,
        run_name="gkd_math_validation",
    )

    trainer = AtlasGKDTrainer(
        model=model,
        teacher_model=teacher_model,
        args=gkd_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    train_start = time.perf_counter()
    trainer_output = trainer.train()
    train_duration = time.perf_counter() - train_start
    trainer.save_model(args.output_dir)

    print(f"Training finished in {train_duration:.2f}s (global_step={trainer.state.global_step}).")

    if not args.no_baseline_eval:
        print("Evaluating distilled student...")
        metrics_summary["distilled"] = evaluate_model(
            model=model,
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            dataset=eval_subset,
            max_new_tokens=args.max_new_tokens,
        )

    metrics_summary["training"] = {
        "duration_seconds": train_duration,
        "global_step": trainer.state.global_step,
        "samples_per_second": trainer_output.metrics.get("train/samples_per_second"),
        "train_runtime": trainer_output.metrics.get("train_runtime"),
        "train_loss": trainer_output.metrics.get("train_loss"),
    }

    summary_path = Path(args.output_dir) / "math_validation_metrics.json"
    summary_path.write_text(json.dumps(metrics_summary, indent=2))
    print(f"Saved metrics summary to {summary_path}")


def evaluate_model(
    model: AutoModelForCausalLM,
    teacher_model: AutoModelForCausalLM,
    tokenizer,
    dataset: Dataset,
    max_new_tokens: int,
) -> Dict[str, float]:
    model.eval()
    teacher_model.eval()

    accuracies: List[int] = []
    token_counts: List[int] = []
    prompt_token_counts: List[int] = []
    reverse_kls: List[float] = []

    for example in dataset:
        prompt_text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
        )
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        gen_outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        sequence = gen_outputs.sequences[0]
        generated_ids = sequence[prompt_len:]
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        predicted_answer = extract_answer_from_text(decoded)
        expected_answer = normalize_math_answer(example.get("final_answer", ""))

        accuracies.append(int(predicted_answer == expected_answer and expected_answer != ""))
        token_counts.append(max(int(generated_ids.shape[0]), 1))
        prompt_token_counts.append(prompt_len)

        reverse_kls.append(
            float(
                compute_reverse_kl(
                    student=model,
                    teacher=teacher_model,
                    sequence=sequence,
                    prompt_length=prompt_len,
                )
            )
        )

    return {
        "accuracy": sum(accuracies) / max(len(accuracies), 1),
        "avg_generated_tokens": sum(token_counts) / max(len(token_counts), 1),
        "avg_prompt_tokens": sum(prompt_token_counts) / max(len(prompt_token_counts), 1),
        "mean_reverse_kl": sum(reverse_kls) / max(len(reverse_kls), 1),
    }


def compute_reverse_kl(
    student: AutoModelForCausalLM,
    teacher: AutoModelForCausalLM,
    sequence: torch.Tensor,
    prompt_length: int,
) -> torch.Tensor:
    """Approximate reverse KL on generated tokens for a single sequence."""
    if sequence.dim() == 1:
        sequence = sequence.unsqueeze(0)

    student_device = next(student.parameters()).device
    teacher_device = next(teacher.parameters()).device

    student_sequence = sequence.to(student_device)
    teacher_sequence = sequence.to(teacher_device)

    student_log_probs = _gather_token_log_probs(student, student_sequence).to(student_device)
    teacher_log_probs = _gather_token_log_probs(teacher, teacher_sequence).to(teacher_device)

    # Only consider generated region
    student_slice = student_log_probs[:, prompt_length - 1 : -1]
    teacher_slice = teacher_log_probs[:, prompt_length - 1 : -1]

    if student_slice.numel() == 0:
        return torch.tensor(0.0, device=student_device)

    reverse_kl = (student_slice - teacher_slice.to(student_device)).sum(dim=-1) / student_slice.shape[-1]
    return reverse_kl.mean()


def _gather_token_log_probs(model: AutoModelForCausalLM, input_ids: torch.Tensor) -> torch.Tensor:
    """Compute per-token log probabilities for next-token predictions."""
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        labels = input_ids[:, 1:].unsqueeze(-1)
        token_log_probs = log_probs[:, :-1, :].gather(-1, labels).squeeze(-1)
    return token_log_probs


if __name__ == "__main__":
    main()
