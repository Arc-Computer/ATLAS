"""Execute ArcOps-Cyber questions in batch for baseline (Student) or guided (Teacher) modes."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Optional, Tuple

from copy import deepcopy

REPO_ROOT = Path(__file__).resolve().parents[2]
SDK_PATH = REPO_ROOT / "external" / "atlas-sdk"
if SDK_PATH.exists():
    sys.path.insert(0, str(SDK_PATH))

from atlas import arun  # type: ignore
from atlas.runtime.orchestration.execution_context import ExecutionContext  # type: ignore

from scripts.arcops_cyber.scoring import score_answer


def _load_env_file():
    env_path = REPO_ROOT / '.env'
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_env_file()

CONFIGS = {
    "student": "configs/examples/arcops_cyber_student.yaml",
    "teacher": "configs/examples/arcops_cyber_runtime.yaml",
}

SCENARIO_PATH = REPO_ROOT / "paper_assets/arcops_cyber/scenario_splits.json"
TASK_ROOT = REPO_ROOT / "paper_assets/arcops_cyber/tasks"


@dataclass
class RunRecord:
    scenario: str
    task_id: str
    incident: str
    question: str
    session_reward_score: Optional[float]
    session_reward: Optional[dict]
    audit_success: Optional[float]
    audit_rationale: Optional[str]
    final_answer: str
    latency_ms: Optional[float]
    tokens_total: Optional[int]
    teacher_guidance_tokens: Optional[int]
    raw_metadata: dict

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "task_id": self.task_id,
            "incident": self.incident,
            "question": self.question,
            "session_reward_score": self.session_reward_score,
            "session_reward": self.session_reward,
            "audit_success": self.audit_success,
            "audit_rationale": self.audit_rationale,
            "final_answer": self.final_answer,
            "latency_ms": self.latency_ms,
            "tokens_total": self.tokens_total,
            "teacher_guidance_tokens": self.teacher_guidance_tokens,
            "metadata": self.raw_metadata,
        }


def load_scenarios(selections: Iterable[str]) -> List[tuple[str, dict]]:
    payload = json.loads(SCENARIO_PATH.read_text())
    scenarios: List[tuple[str, dict]] = []
    for key in selections:
        if key not in payload:
            raise ValueError(f"Unknown scenario '{key}'")
        entry = payload[key]
        entry.setdefault("label", key)
        scenarios.append((key, entry))
    return scenarios


def build_task_prompt(task_path: Path) -> dict:
    data = json.loads((TASK_ROOT / task_path).read_text())
    context = data.get("context", "").strip()
    question = data.get("question", "").strip()
    prompt = (
        "You are assisting with a cyber threat investigation. Use only the context provided to answer.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    return {
        "task_id": data["task_id"],
        "incident": data["incident"],
        "question": question,
        "gold_answer": data["gold_answer"],
        "prompt": prompt,
    }


def extract_metrics(step_metadata: dict) -> tuple[Optional[float], Optional[int], Optional[int]]:
    runtime = step_metadata.get("runtime") if isinstance(step_metadata, dict) else None
    latency_ms = None
    if isinstance(runtime, dict):
        latency_ms = runtime.get("timings_ms", {}).get("total_ms")
    usage = step_metadata.get("usage") if isinstance(step_metadata, dict) else None
    tokens_total = None
    teacher_tokens = None
    if isinstance(usage, dict):
        total = usage.get("total_tokens")
        if isinstance(total, (int, float)):
            tokens_total = int(total)
        teacher_usage = usage.get("teacher")
        if isinstance(teacher_usage, dict):
            teacher_total = teacher_usage.get("total_tokens")
            if isinstance(teacher_total, (int, float)):
                teacher_tokens = int(teacher_total)
    return latency_ms, tokens_total, teacher_tokens


def _avg(values: List[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    return mean(vals) if vals else None


def _summarise(records: List[RunRecord]) -> dict:
    by_scenario: dict[str, List[RunRecord]] = defaultdict(list)
    for record in records:
        by_scenario[record.scenario].append(record)

    def build_summary(rows: List[RunRecord]) -> dict:
        return {
            "count": len(rows),
            "avg_reward_score": _avg([r.session_reward_score for r in rows]),
            "avg_latency_ms": _avg([r.latency_ms for r in rows]),
            "avg_tokens_total": _avg([r.tokens_total for r in rows]),
            "avg_teacher_tokens": _avg([r.teacher_guidance_tokens for r in rows]),
            "audit_success_rate": _avg([r.audit_success for r in rows]),
        }

    scenario_summary = {
        scenario: build_summary(rows)
        for scenario, rows in by_scenario.items()
    }
    overall = build_summary(records) if records else {"count": 0}
    return {"overall": overall, "scenarios": scenario_summary}


async def _run_single_task(prompt: str, config_path: str) -> tuple:
    """Execute a single task via the async Atlas runtime and capture metadata."""
    context = ExecutionContext.get()
    context.reset()
    result = await arun(
        task=prompt,
        config_path=config_path,
        stream_progress=False,
    )
    metadata = deepcopy(context.metadata)
    context.reset()
    return result, metadata


def _normalise_reward_payload(payload: object) -> tuple[Optional[float], Optional[dict]]:
    if payload is None:
        return None, None
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()  # type: ignore[assignment, union-attr]
    if isinstance(payload, dict):
        score = payload.get("score")
        try:
            score_val = float(score) if score is not None else None
        except (TypeError, ValueError):
            score_val = None
        return score_val, payload
    return None, None


async def _run_batch_async(mode: str, scenarios: List[tuple[str, dict]], limit: Optional[int] = None) -> Tuple[List[RunRecord], dict]:
    config_path = CONFIGS[mode]
    results: List[RunRecord] = []
    for label, scenario in scenarios:
        name = scenario.get("label", label)
        incident = scenario["incident"]
        for q in scenario["questions"]:
            task_meta = build_task_prompt(Path(q["context_path"]))
            result, metadata = await _run_single_task(task_meta["prompt"], config_path)

            reward_summary = metadata.get("session_reward") or metadata.get("reward_summary")
            reward_score, reward_payload = _normalise_reward_payload(reward_summary)

            audit_score = audit_rationale = None
            if task_meta.get("gold_answer"):
                audit_score, audit_rationale = score_answer(task_meta["gold_answer"], result.final_answer)

            step_metadata = result.step_results[0].metadata if result.step_results else {}
            latency_ms, tokens_total, teacher_tokens = extract_metrics(step_metadata)
            record = RunRecord(
                scenario=name,
                task_id=task_meta["task_id"],
                incident=incident,
                question=task_meta["question"],
                session_reward_score=reward_score,
                session_reward=reward_payload,
                audit_success=audit_score,
                audit_rationale=audit_rationale,
                final_answer=result.final_answer,
                latency_ms=latency_ms,
                tokens_total=tokens_total,
                teacher_guidance_tokens=teacher_tokens,
                raw_metadata=step_metadata,
            )
            results.append(record)
            if limit is not None and len(results) >= limit:
                summary = _summarise(results)
                return results, summary
    summary = _summarise(results)
    return results, summary


def run_batch(mode: str, scenarios: List[tuple[str, dict]], limit: Optional[int] = None) -> Tuple[List[RunRecord], dict]:
    return asyncio.run(_run_batch_async(mode, scenarios, limit))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ArcOps-Cyber batches")
    parser.add_argument("mode", choices=CONFIGS.keys(), help="Execution mode: student or teacher")
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=["scenario_1", "scenario_2", "scenario_3", "scenario_4", "scenario_5"],
        help="Scenario keys to run (defaults to all)",
    )
    parser.add_argument("--limit", type=int, help="Stop after N questions across all scenarios")
    parser.add_argument("--output", type=Path, default=Path("paper_assets/arcops_cyber/results.json"))
    args = parser.parse_args()

    selected = load_scenarios(args.scenarios)
    records, summary = run_batch(args.mode, selected, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"mode": args.mode, "records": [r.to_dict() for r in records], "summary": summary}
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {len(records)} records to {args.output}")
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
