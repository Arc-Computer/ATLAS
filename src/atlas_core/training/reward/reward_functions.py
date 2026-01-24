"""Reward functions for GRPO experiments."""

from __future__ import annotations

from typing import List

from atlas_core.data.math_gkd import extract_answer_from_text, normalize_math_answer


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


def trace_reward_from_score(
    prompts: List[str],
    completions: List[str],
    reward_score: List[float] | None = None,
    **_: dict,
) -> List[float]:
    """Use reward scores embedded in runtime traces as the GRPO signal."""

    if reward_score is None:
        return [0.0 for _ in prompts]
    rewards: List[float] = []
    for score in reward_score:
        try:
            rewards.append(float(score))
        except (TypeError, ValueError):
            rewards.append(0.0)
    return rewards
