"""Utilities for consuming runtime JSONL exports in offline trainers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, List, Dict, Any, Sequence

from atlas_core.runtime import (
    AtlasRewardBreakdown,
    AtlasJudgeBreakdown,
    AtlasJudgeSample,
    AtlasSessionTrace,
    AtlasStepTrace,
)


def _coerce_reward(entry: Dict[str, Any]) -> AtlasRewardBreakdown:
    judges_data: Sequence[Dict[str, Any]] = entry.get("judges") or []
    judges: List[AtlasJudgeBreakdown] = []
    for judge in judges_data:
        samples_payload = judge.get("samples") or []
        samples = [
            AtlasJudgeSample(
                score=sample.get("score", 0.0),
                rationale=sample.get("rationale", ""),
                principles=sample.get("principles", []) or [],
                uncertainty=sample.get("uncertainty"),
                temperature=sample.get("temperature"),
            )
            for sample in samples_payload
        ]
        judges.append(
            AtlasJudgeBreakdown(
                identifier=judge.get("identifier", ""),
                score=judge.get("score", 0.0),
                rationale=judge.get("rationale", ""),
                principles=judge.get("principles", []) or [],
                samples=samples,
                escalated=bool(judge.get("escalated", False)),
                escalation_reason=judge.get("escalation_reason"),
            )
        )
    score = entry.get("score", 0.0)
    rationale = entry.get("rationale")
    raw_payload = entry.get("raw")
    return AtlasRewardBreakdown(
        score=score,
        judges=judges,
        rationale=rationale,
        raw=raw_payload if isinstance(raw_payload, dict) else entry,
    )


def _coerce_step(step: Dict[str, Any]) -> AtlasStepTrace:
    reward_payload = step.get("evaluation") or step.get("reward") or {}
    reward = (
        reward_payload
        if isinstance(reward_payload, AtlasRewardBreakdown)
        else _coerce_reward(reward_payload)
    )
    raw_context = step.get("context") or step.get("prior_results") or step.get("context_outputs")
    if isinstance(raw_context, str):
        try:
            context = json.loads(raw_context)
        except json.JSONDecodeError:
            context = {"raw": raw_context}
    else:
        context = raw_context or {}

    raw_tool_params = step.get("tool_params") or {}
    if isinstance(raw_tool_params, str):
        try:
            tool_params = json.loads(raw_tool_params)
        except json.JSONDecodeError:
            tool_params = {"raw": raw_tool_params}
    else:
        tool_params = raw_tool_params or {}

    tool_name = step.get("tool") or tool_params.get("name")
    return AtlasStepTrace(
        step_id=step.get("step_id", step.get("id", 0)),
        description=step.get("description", ""),
        trace=step.get("trace", ""),
        output=step.get("output", ""),
        reward=reward,
        tool=tool_name,
        tool_params=tool_params,
        context=context if isinstance(context, dict) else {},
        validation=step.get("validation", {}),
        attempts=step.get("attempts", 1),
        guidance=step.get("guidance", []) or [],
        metadata={
            key: value
            for key, value in step.items()
            if key
            not in {
                "step_id",
                "id",
                "description",
                "trace",
                "output",
                "evaluation",
                "reward",
                "validation",
                "attempts",
                "guidance",
                "context",
                "context_outputs",
                "prior_results",
                "tool",
                "tool_params",
            }
        },
    )


def _coerce_session(record: Dict[str, Any]) -> AtlasSessionTrace:
    steps_field = record.get("steps") or []
    steps = [_coerce_step(step) for step in steps_field]
    return AtlasSessionTrace(
        task=record.get("task", ""),
        final_answer=record.get("final_answer", ""),
        plan=record.get("plan", {}),
        steps=steps,
        session_metadata=record.get("session_metadata", {}) or {},
    )


def load_runtime_traces(path: str | Path) -> List[AtlasSessionTrace]:
    """Load runtime traces exported as JSONL into structured dataclasses."""

    path = Path(path)
    sessions: List[AtlasSessionTrace] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            sessions.append(_coerce_session(payload))
    return sessions


def flatten_traces_for_training(
    sessions: Iterable[AtlasSessionTrace],
) -> List[Dict[str, Any]]:
    """Flatten session traces into trainer-friendly dicts."""

    records: List[Dict[str, Any]] = []
    for session in sessions:
        for step in session.steps:
            records.append(
                {
                    "task": session.task,
                    "final_answer": session.final_answer,
                    "plan": session.plan,
                    "step_id": step.step_id,
                    "step_description": step.description,
                    "step_trace": step.trace,
                    "step_output": step.output,
                    "attempts": step.attempts,
                    "guidance_history": step.guidance,
                    "step_context": step.context,
                    "validation": step.validation,
                    "tool": step.tool,
                    "tool_params": step.tool_params,
                    "reward_score": step.reward.score,
                    "reward_breakdown": step.reward.to_dict(),
                    "session_metadata": session.session_metadata,
                    "step_metadata": step.metadata,
                }
            )
    return records


def build_executor_prompt(step: AtlasStepTrace, session: AtlasSessionTrace) -> str:
    """Create a textual prompt mirroring the runtime executor input."""

    system_prompt = (
        step.metadata.get("executor_system_prompt")
        or session.session_metadata.get("executor_system_prompt")
        or "You are the Atlas Student executor. Follow the step instructions precisely."
    )
    context_block = json.dumps(step.context or {}, ensure_ascii=False, indent=2)
    guidance_block = json.dumps(step.guidance or [], ensure_ascii=False, indent=2)
    validation_block = json.dumps(step.validation or {}, ensure_ascii=False, indent=2)
    payload = [
        f"Task: {session.task}",
        f"Step ID: {step.step_id}",
        f"Description: {step.description}",
        f"Tool: {step.tool or 'none'}",
        f"Tool Parameters: {json.dumps(step.tool_params or {}, ensure_ascii=False)}",
        f"Context: {context_block}",
        f"Guidance History: {guidance_block}",
        f"Previous Validation: {validation_block}",
    ]
    user_message = "\n".join(payload)
    return f"{system_prompt}\n\n{user_message}"


def sessions_to_rl_records(
    sessions: Iterable[AtlasSessionTrace],
) -> List[Dict[str, Any]]:
    """Convert runtime sessions into RL-ready dataset records."""

    records: List[Dict[str, Any]] = []
    for session in sessions:
        for step in session.steps:
            prompt_text = build_executor_prompt(step, session)
            record = {
                "prompt": prompt_text,
                "task": session.task,
                "plan": session.plan,
                "step_id": step.step_id,
                "step_description": step.description,
                "guidance_history": step.guidance,
                "step_context": step.context,
                "validation": step.validation,
                "tool": step.tool,
                "tool_params": step.tool_params,
                "reward_breakdown": step.reward.to_dict(),
                "reward_score": step.reward.score,
                "step_output": step.output,
                "step_trace": step.trace,
                "session_metadata": session.session_metadata,
            }
            ground_truth = step.metadata.get("ground_truth") or session.session_metadata.get("ground_truth")
            if ground_truth:
                record["ground_truth"] = ground_truth
            records.append(record)
    return records


def iter_reward_scores(sessions: Iterable[AtlasSessionTrace]) -> Iterator[float]:
    """Yield reward scores from each step in the provided sessions."""

    for session in sessions:
        for step in session.steps:
            yield step.reward.score
