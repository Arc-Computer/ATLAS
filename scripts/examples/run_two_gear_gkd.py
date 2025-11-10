#!/usr/bin/env python3
"""
Launch the two-gear GSM8K GKD experiments back-to-back and summarize the results.

This helper reproduces the commands documented in gkd_two_gear_gkd_blog_draft.md:

- Gear 1 (diagnostic): 500 steps, higher LR / temperature, full GSM8K splits.
- Gear 2 (reliability): 2,500 steps, lower LR / temperature, tighter generation.

Each run writes into its own output directory to avoid metric churn, and the script
prints a comparison table (train/eval loss, accuracy deltas, token deltas) once both
jobs finish. WandB logging is preserved by inheriting the caller's environment.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
VALIDATE_SCRIPT = ROOT / "scripts" / "validate_gkd.py"


@dataclass
class GearRun:
    name: str
    output_dir: Path
    extra_env: Dict[str, str] = field(default_factory=dict)
    args: List[str] = field(default_factory=list)


def run_cmd(run: GearRun) -> None:
    output_dir = run.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(VALIDATE_SCRIPT),
        "--output-dir",
        str(output_dir),
    ] + run.args

    env = os.environ.copy()
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    env.setdefault("PYTHONPATH", str(ROOT))
    env.update(run.extra_env)

    print(f"\n==== Launching {run.name} ====")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def load_metrics(output_dir: Path) -> Dict:
    metrics_path = output_dir / "math_validation_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file missing at {metrics_path}")
    return json.loads(metrics_path.read_text())


def summarize(metrics: Dict) -> Dict[str, float]:
    baseline = metrics.get("baseline", {})
    distilled = metrics.get("distilled", {})
    summary = {
        "train_loss": metrics.get("training", {}).get("train_loss"),
        "eval_accuracy": distilled.get("accuracy"),
        "avg_tokens": distilled.get("avg_generated_tokens"),
    }
    if baseline and distilled:
        baseline_acc = baseline.get("accuracy")
        distilled_acc = distilled.get("accuracy")
        if baseline_acc is not None and distilled_acc is not None:
            summary["success_delta"] = distilled_acc - baseline_acc
        baseline_tokens = baseline.get("avg_generated_tokens")
        distilled_tokens = distilled.get("avg_generated_tokens")
        if baseline_tokens and distilled_tokens:
            summary["token_reduction_pct"] = 1.0 - (distilled_tokens / baseline_tokens)
    return summary


def print_comparison(results: List[Tuple[str, Dict]]) -> None:
    header = f"{'Run':<12} {'Train Loss':>12} {'Eval Acc':>12} {'Δ Success':>12} {'Token Δ%':>12}"
    print("\n==== Final Comparison ====")
    print(header)
    print("-" * len(header))
    for name, summary in results:
        train_loss = summary.get("train_loss")
        eval_accuracy = summary.get("eval_accuracy")
        success_delta = summary.get("success_delta")
        token_delta = summary.get("token_reduction_pct")
        train_loss = float("nan") if train_loss is None else train_loss
        eval_accuracy = float("nan") if eval_accuracy is None else eval_accuracy
        success_delta = float("nan") if success_delta is None else success_delta
        token_delta = float("nan") if token_delta is None else token_delta * 100
        print(
            f"{name:<12} "
            f"{train_loss:>12.4f} "
            f"{eval_accuracy:>12.4f} "
            f"{success_delta:>12.4f} "
            f"{token_delta:>11.2f}%"
        )


def main() -> None:
    common_args = [
        "--student",
        "Qwen/Qwen2.5-7B-Instruct",
        "--teacher",
        "Qwen/Qwen2.5-14B-Instruct",
        "--dataset-name",
        "gsm8k",
        "--dataset-config",
        "main",
        "--dataset-train-split",
        "train",
        "--dataset-eval-split",
        "test",
        "--dataset-max-samples",
        "8792",
        "--train-limit",
        "7473",
        "--eval-limit",
        "1319",
        "--per-device-train-batch-size",
        "2",
        "--gradient-accumulation-steps",
        "4",
        "--eval-sample-size",
        "256",
        "--bf16",
    ]

    runs = [
        GearRun(
            name="gear1-diagnostic",
            output_dir=ROOT / "outputs" / "gkd_two_gear" / "gear1",
            args=common_args
            + [
                "--max-steps",
                "500",
                "--learning-rate",
                "2e-5",
                "--lmbda",
                "1.0",
                "--beta",
                "0.5",
                "--temperature",
                "0.9",
                "--max-new-tokens",
                "256",
            ],
        ),
        GearRun(
            name="gear2-reliability",
            output_dir=ROOT / "outputs" / "gkd_two_gear" / "gear2",
            args=common_args
            + [
                "--max-steps",
                "2500",
                "--learning-rate",
                "3e-6",
                "--lmbda",
                "1.0",
                "--beta",
                "0.5",
                "--temperature",
                "0.6",
                "--max-new-tokens",
                "128",
            ],
        ),
    ]

    results: List[Tuple[str, Dict]] = []
    for run in runs:
        run_cmd(run)
        metrics = load_metrics(run.output_dir)
        summary = summarize(metrics)
        results.append((run.name, summary))

    print_comparison(results)


if __name__ == "__main__":
    main()
