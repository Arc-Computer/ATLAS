"""Reward functions for GRPO experiments."""

from __future__ import annotations

from typing import List

from trainers.math_gkd_dataset import extract_answer_from_text, normalize_math_answer


def meta_math_exact_match_reward(
    prompts: List[str],
    completions: List[str],
    ground_truth: List[str],
    **_: dict,
) -> List[float]:
    """Binary reward: 1 if student's answer matches MetaMathQA ground truth."""

    rewards: List[float] = []
    for completion, target in zip(completions, ground_truth):
        predicted = normalize_math_answer(extract_answer_from_text(completion or ""))
        expected = normalize_math_answer(target or "")
        rewards.append(1.0 if predicted and predicted == expected else 0.0)
    return rewards
