#!/usr/bin/env python
"""Quick evaluation flow: baseline vs teacher-guided response scored with RIM."""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Sequence

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from openai import OpenAI

from RIM.reward_adapter import RIMReward


DEFAULT_QUESTION = (
    "Masha braided her dolls' hair: half received one braid, a quarter received two, "
    "and the remaining quarter received four. She used 24 ribbons total. How many dolls does she have?"
)


def extract_text(response) -> str:
    """Return best-effort text from a Responses API result."""
    chunks: List[str] = []
    output = getattr(response, "output", None)
    if isinstance(output, list):
        for item in output:
            if getattr(item, "type", None) == "output_text":
                chunks.append(getattr(item, "text", ""))
            elif hasattr(item, "content"):  # compatibility
                for content in getattr(item, "content", []) or []:
                    if getattr(content, "type", None) == "output_text":
                        chunks.append(getattr(content, "text", ""))
    if chunks:
        return "".join(chunks).strip()
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text.strip()
    return str(response)


def call_responses_api(
    client: OpenAI,
    model: str,
    messages: Sequence[dict],
    max_output_tokens: int | None = None,
) -> str:
    kwargs = {
        "model": model,
        "input": list(messages),
    }
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    response = client.responses.create(**kwargs)
    return extract_text(response)


def evaluate_question(
    question: str,
    teacher_model: str,
    student_model: str,
    reward: RIMReward,
    client: OpenAI,
    teacher_tokens: int | None,
    student_tokens: int | None,
    show_rationales: bool,
) -> None:
    baseline_messages = [
        {
            "role": "system",
            "content": "You are a helpful agent. Tackle the task carefully and explain your reasoning.",
        },
        {"role": "user", "content": question},
    ]
    baseline = call_responses_api(
        client,
        model=student_model,
        messages=baseline_messages,
        max_output_tokens=student_tokens,
    )

    teacher_messages = [
        {
            "role": "system",
            "content": (
                "You are an expert teacher. Read the student's attempt and provide concise, actionable teaching. "
                "Always wrap guidance in <teaching> tags."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Student attempt:\n{baseline}\n\n"
                "Point out what to fix, why it matters, and how to improve, using <teaching>...</teaching>."
            ),
        },
    ]
    teaching = call_responses_api(
        client,
        model=teacher_model,
        messages=teacher_messages,
        max_output_tokens=teacher_tokens,
    )

    enhanced_messages = [
        {
            "role": "system",
            "content": "You are a diligent student. Apply the provided teaching step by step.",
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Teaching provided:\n{teaching}\n\n"
                "Use the teaching to solve the problem. Show reasoning and put the final answer in <solution> tags."
            ),
        },
    ]
    enhanced = call_responses_api(
        client,
        model=student_model,
        messages=enhanced_messages,
        max_output_tokens=student_tokens,
    )

    baseline_eval = reward.evaluate(prompt=question, response=baseline)
    enhanced_eval = reward.evaluate(
        prompt=question,
        response=enhanced,
        baseline_solutions=baseline,
        teacher_traces=teaching,
    )

    divider = "=" * 72
    print(divider)
    print("Baseline student answer:\n" + baseline.strip() + "\n")
    print("Teacher guidance:\n" + teaching.strip() + "\n")
    print("Student with teaching:\n" + enhanced.strip() + "\n")
    print(divider)
    print(f"Reward (baseline): {baseline_eval.score:.3f}")
    if baseline_eval.judge_scores:
        print(f"Per-judge (baseline): {baseline_eval.judge_scores}")
    if show_rationales and baseline_eval.rationale:
        print(f"Rationale (baseline):\n{baseline_eval.rationale}\n")

    print(f"Reward (with teaching): {enhanced_eval.score:.3f}")
    if enhanced_eval.judge_scores:
        print(f"Per-judge (with teaching): {enhanced_eval.judge_scores}")
    if show_rationales and enhanced_eval.rationale:
        print(f"Rationale (with teaching):\n{enhanced_eval.rationale}\n")

    print(f"Delta: {enhanced_eval.score - baseline_eval.score:+.3f}")
    print(divider)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick ATLAS evaluation loop")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Question to evaluate")
    parser.add_argument("--teacher-model", default="gpt-5", help="Model id for the teacher")
    parser.add_argument(
        "--student-model",
        default="gpt-4o-mini",
        help="Model id for the student",
    )
    parser.add_argument(
        "--reward-config",
        default="configs/rim_config.yaml",
        help="Path to RIM reward configuration",
    )
    parser.add_argument(
        "--teacher-max-output",
        type=int,
        default=512,
        help="Max tokens for the teacher response (use values supported by the model)",
    )
    parser.add_argument(
        "--student-max-output",
        type=int,
        default=512,
        help="Max tokens for student generations",
    )
    parser.add_argument(
        "--verbose-judges",
        action="store_true",
        help="Print full judge model call traces (default: off)",
    )
    args = parser.parse_args()

    os.environ["RIM_VERBOSE"] = "1" if args.verbose_judges else "0"

    client = OpenAI()
    reward = RIMReward(config_path=args.reward_config)

    evaluate_question(
        question=args.question,
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        reward=reward,
        client=client,
        teacher_tokens=args.teacher_max_output,
        student_tokens=args.student_max_output,
        show_rationales=args.verbose_judges,
    )


if __name__ == "__main__":
    main()
