import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainers.runtime_dataset import (
    load_runtime_traces,
    flatten_traces_for_training,
    build_executor_prompt,
    sessions_to_rl_records,
)


def test_load_runtime_traces_round_trip(tmp_path: Path):
    payload = {
        "task": "Solve problem A",
        "final_answer": "answer",
        "plan": {"steps": [{"id": 1, "description": "step"}]},
        "session_metadata": {"dataset": "demo"},
        "steps": [
            {
                "step_id": 1,
                "description": "Compute baseline",
                "trace": "HUMAN: do work",
                "output": "result",
                "evaluation": {
                    "score": 0.9,
                    "rationale": "Good job",
                    "judges": [
                        {
                            "identifier": "accuracy",
                            "score": 0.9,
                            "rationale": "Matches reference",
                            "principles": [{"name": "Accuracy", "weight": 1.0}],
                            "samples": [
                                {
                                    "score": 0.9,
                                    "rationale": "small-model",
                                    "principles": [],
                                    "uncertainty": 0.1,
                                    "temperature": 0.2,
                                }
                            ],
                            "escalated": False,
                            "escalation_reason": None,
                        }
                    ],
                },
                "validation": {"valid": True},
                "attempts": 1,
                "guidance": ["keep going"],
            }
        ],
    }
    path = tmp_path / "export.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    sessions = load_runtime_traces(path)
    assert len(sessions) == 1
    session = sessions[0]
    assert session.task == "Solve problem A"
    assert session.steps[0].reward.score == 0.9
    assert session.steps[0].reward.judges[0].identifier == "accuracy"
    assert session.steps[0].reward.judges[0].samples[0].temperature == 0.2

    flattened = flatten_traces_for_training(sessions)
    assert len(flattened) == 1
    record = flattened[0]
    assert record["task"] == "Solve problem A"
    assert record["reward_score"] == 0.9
    assert record["reward_breakdown"]["judges"][0]["identifier"] == "accuracy"


def test_sessions_to_rl_records_builds_prompt():
    from atlas_core.runtime import AtlasSessionTrace, AtlasStepTrace, AtlasRewardBreakdown

    reward = AtlasRewardBreakdown(score=0.8, judges=[], rationale="check", raw={"score": 0.8})
    step = AtlasStepTrace(
        step_id=1,
        description="Retrieve baseline metrics",
        trace="HUMAN: Step ID 1 ...",
        output="baseline response",
        reward=reward,
        guidance=["Use cautious reasoning"],
        validation={"valid": True},
        context={"0": "prior output"},
        tool="search",
        tool_params={"query": "metrics"},
        metadata={"executor_system_prompt": "You are the runtime executor."},
    )
    session = AtlasSessionTrace(
        task="Diagnose outage",
        final_answer="Resolved issue",
        plan={"steps": [{"id": 1, "description": "Retrieve baseline metrics"}]},
        steps=[step],
        session_metadata={},
    )

    prompt_text = build_executor_prompt(step, session)
    assert "Diagnose outage" in prompt_text
    assert "Step ID: 1" in prompt_text
    assert "Guidance History" in prompt_text

    records = sessions_to_rl_records([session])
    assert records[0]["prompt"] == prompt_text
    assert records[0]["reward_breakdown"]["score"] == 0.8
