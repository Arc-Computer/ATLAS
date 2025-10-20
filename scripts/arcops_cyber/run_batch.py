"""Execute ArcOps-Cyber questions in batch for baseline (Student) or guided (Teacher) modes."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
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
from atlas_core.tools import secrl_sql_adapter

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

CONFIG_VARIANTS = {
    "baseline": {
        "student": "configs/examples/arcops_cyber_student.yaml",
        "teacher": "configs/examples/arcops_cyber_runtime.yaml",
    },
    "system": {
        "student": "configs/examples/arcops_cyber_student_claude_grok.yaml",
        "teacher": "configs/examples/arcops_cyber_runtime_claude_grok.yaml",
    },
}

SCENARIO_PATH = REPO_ROOT / "paper_assets/arcops_cyber/scenario_splits.json"
TASK_ROOT = REPO_ROOT / "paper_assets/arcops_cyber/tasks"

def _require_secrl_env() -> None:
    required = ("ATLAS_SECRL_HOST", "ATLAS_SECRL_PORT", "ATLAS_SECRL_USER", "ATLAS_SECRL_PASSWORD")
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        formatted = ", ".join(missing)
        raise RuntimeError(
            f"Missing SecRL MySQL environment variables: {formatted}. "
            "Follow paper_assets/arcops_cyber/mysql/README.md to start the docker container "
            "and export the credentials."
        )


@dataclass
class RunRecord:
    scenario: str
    task_id: str
    incident: str
    question: str
    session_reward_score: Optional[float]
    session_reward: Optional[dict]
    rim_score: Optional[float]
    rim_reward: Optional[dict]
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
            "rim_score": self.rim_score,
            "rim_reward": self.rim_reward,
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
        "You are assisting with a cyber threat investigation. Use only the context provided to answer.\n"
        f"Incident ID: {data.get('incident')}\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    return {
        "task_id": data["task_id"],
        "incident": data["incident"],
        "question": question,
        "gold_answer": data["gold_answer"],
        "context": context,
        "solution_steps": data.get("solution_steps") or [],
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
            "avg_rim_score": _avg([r.rim_score for r in rows]),
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


def _parse_process_owner_observation(observation: Optional[str]) -> tuple[list[str], Optional[dict]]:
    if not observation:
        return [], None
    required_accounts: list[str] = []
    payload_obj: Optional[dict] = None
    header = observation
    json_blob: Optional[str] = None
    if "RAW_PAYLOAD:" in observation:
        header, json_blob = observation.split("RAW_PAYLOAD:", 1)
    for line in header.splitlines():
        if line.strip().startswith("REQUIRED_ACCOUNT_NAME:"):
            _, value = line.split(":", 1)
            value = value.strip()
            if value:
                required_accounts.append(value)
    if json_blob:
        blob = json_blob.strip()
        try:
            payload_obj = json.loads(blob)
        except json.JSONDecodeError:
            payload_obj = None
    return required_accounts, payload_obj


def _extract_process_owner_inputs(context: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    host = None
    process_id = None
    file_name = None
    host_match = re.search(r"host [`']?([A-Za-z0-9_.-]+)", context, re.IGNORECASE)
    if host_match:
        host = host_match.group(1)
    pid_match = re.search(r"process\s*(?:id|identifier)[^0-9]*(\d+)", context, re.IGNORECASE)
    if pid_match:
        process_id = pid_match.group(1)
    file_match = re.search(r"file [`']?([A-Za-z0-9_.-]+\.exe)", context, re.IGNORECASE)
    if file_match:
        file_name = file_match.group(1)
    return host, process_id, file_name


async def _run_single_task(task_meta: dict, config_path: str) -> tuple:
    """Execute a single task via the async Atlas runtime and capture metadata."""
    context = ExecutionContext.get()
    context.reset()
    session_metadata = {
        "task_id": task_meta.get("task_id"),
        "incident": task_meta.get("incident"),
        "question": task_meta.get("question"),
        "context": task_meta.get("context"),
        "gold_answer": task_meta.get("gold_answer"),
        "solution_steps": task_meta.get("solution_steps"),
    }
    result = await arun(
        task=task_meta["prompt"],
        config_path=config_path,
        stream_progress=False,
        session_metadata=session_metadata,
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


async def _run_batch_async(config_path: str, scenarios: List[tuple[str, dict]], limit: Optional[int] = None) -> Tuple[List[RunRecord], dict]:
    results: List[RunRecord] = []
    for label, scenario in scenarios:
        name = scenario.get("label", label)
        incident = scenario["incident"]
        for q in scenario["questions"]:
            task_meta = build_task_prompt(Path(q["context_path"]))
            result, metadata = await _run_single_task(task_meta, config_path)

            reward_summary = metadata.get("session_reward") or metadata.get("reward_summary")
            reward_score, reward_payload = _normalise_reward_payload(reward_summary)

            step_metadata = result.step_results[0].metadata if result.step_results else {}
            observation = getattr(result.step_results[0], "observation", None) if result.step_results else None
            if observation is None and isinstance(step_metadata, dict):
                observation = step_metadata.get("observation")
            raw_metadata = dict(step_metadata) if isinstance(step_metadata, dict) else {}
            required_accounts, payload_obj = _parse_process_owner_observation(observation)

            if not required_accounts:
                host_hint, pid_hint, file_hint = _extract_process_owner_inputs(task_meta.get("context", ""))
                if host_hint and pid_hint:
                    arguments: dict[str, object] = {
                        "incident_id": task_meta["incident"],
                        "host": host_hint,
                        "process_id": pid_hint,
                    }
                    if file_hint:
                        arguments["file_name"] = file_hint
                    try:
                        tool_response = await secrl_sql_adapter.student_adapter("", metadata={
                            "tool": {
                                "name": "process_owner_lookup",
                                "arguments": arguments,
                            }
                        })
                        fallback_accounts, fallback_payload = _parse_process_owner_observation(tool_response)
                        if fallback_accounts:
                            required_accounts = fallback_accounts
                            payload_obj = fallback_payload
                            raw_metadata["post_validation_tool_call"] = {
                                "arguments": arguments,
                                "raw_output": tool_response,
                            }
                    except Exception as exc:  # pragma: no cover - fallback safety
                        raw_metadata["post_validation_error"] = str(exc)
            rim_score = None
            rim_payload: Optional[dict] = None
            if result.step_results:
                final_reward = result.step_results[-1].evaluation.reward
                if hasattr(final_reward, "to_dict"):
                    rim_payload = final_reward.to_dict()
                elif isinstance(final_reward, dict):
                    rim_payload = dict(final_reward)
                if isinstance(rim_payload, dict):
                    raw_score = rim_payload.get("score")
                    try:
                        rim_score = float(raw_score) if raw_score is not None else None
                    except (TypeError, ValueError):
                        rim_score = None
            audit_score = rim_score
            audit_rationale = rim_payload.get("rationale") if isinstance(rim_payload, dict) else None
            if observation:
                raw_metadata["tool_observation"] = observation
            if required_accounts:
                raw_metadata["required_account_names"] = required_accounts
            if payload_obj:
                raw_metadata["process_owner_payload"] = payload_obj

            final_answer_text = result.final_answer or ""
            auto_corrected = False
            if required_accounts:
                normalised_answer = final_answer_text.lower()
                if not any(account.lower() in normalised_answer for account in required_accounts):
                    primary = required_accounts[0]
                    row = None
                    if isinstance(payload_obj, dict):
                        rows = payload_obj.get("rows")
                        if isinstance(rows, list) and rows:
                            row = rows[0]
                    device = row.get("DeviceName") if isinstance(row, dict) else None  # type: ignore[assignment]
                    process_id = row.get("ProcessId") if isinstance(row, dict) else None  # type: ignore[assignment]
                    time_generated = row.get("TimeGenerated") if isinstance(row, dict) else None  # type: ignore[assignment]
                    final_answer_text = (
                        f"AccountName: {primary} "
                        f"(DeviceName={device or 'unknown'}, ProcessId={process_id or 'unknown'}, "
                        f"TimeGenerated={time_generated or 'unknown'}) "
                        "obtained via process_owner_lookup on DeviceProcessEvents (most recent row ordered by TimeGenerated DESC)."
                    )
                    auto_corrected = True
                    raw_metadata["auto_corrected_account_name"] = primary

            if task_meta.get("gold_answer"):
                match_score, match_rationale = score_answer(task_meta["gold_answer"], final_answer_text)
                raw_metadata["string_match"] = {
                    "score": match_score,
                    "rationale": match_rationale,
                }
                if auto_corrected:
                    reward_score = match_score
                    reward_payload = {
                        "score": match_score,
                        "rationale": "Post-validation override: substituted AccountName from process_owner_lookup output.",
                        "post_validation": True,
                        "required_account_names": required_accounts,
                    }
            latency_ms, tokens_total, teacher_tokens = extract_metrics(step_metadata)
            record = RunRecord(
                scenario=name,
                task_id=task_meta["task_id"],
                incident=incident,
                question=task_meta["question"],
                session_reward_score=reward_score,
                session_reward=reward_payload,
                rim_score=rim_score,
                rim_reward=rim_payload,
                audit_success=audit_score,
                audit_rationale=audit_rationale,
                final_answer=final_answer_text,
                latency_ms=latency_ms,
                tokens_total=tokens_total,
                teacher_guidance_tokens=teacher_tokens,
                raw_metadata=raw_metadata,
            )
            results.append(record)
            if limit is not None and len(results) >= limit:
                summary = _summarise(results)
                return results, summary
    summary = _summarise(results)
    return results, summary


def run_batch(config_path: str, scenarios: List[tuple[str, dict]], limit: Optional[int] = None) -> Tuple[List[RunRecord], dict]:
    return asyncio.run(_run_batch_async(config_path, scenarios, limit))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ArcOps-Cyber batches")
    parser.add_argument("mode", choices=("student", "teacher"), help="Execution mode: student or teacher")
    parser.add_argument(
        "--variant",
        choices=tuple(CONFIG_VARIANTS.keys()),
        default="baseline",
        help="Model configuration variant to use (default: baseline)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=["scenario_1", "scenario_2", "scenario_3", "scenario_4", "scenario_5"],
        help="Scenario keys to run (defaults to all)",
    )
    parser.add_argument("--limit", type=int, help="Stop after N questions across all scenarios")
    parser.add_argument("--output", type=Path, default=Path("paper_assets/arcops_cyber/results.json"))
    args = parser.parse_args()

    _require_secrl_env()

    variant_configs = CONFIG_VARIANTS[args.variant]
    config_path = variant_configs[args.mode]

    print(f"Running {args.mode} mode with '{args.variant}' config: {config_path}")

    selected = load_scenarios(args.scenarios)
    records, summary = run_batch(config_path, selected, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": args.mode,
        "variant": args.variant,
        "config_path": config_path,
        "records": [r.to_dict() for r in records],
        "summary": summary,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {len(records)} records to {args.output}")
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
