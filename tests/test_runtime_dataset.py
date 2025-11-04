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


def test_load_traces_preserves_sdk_fields(tmp_path: Path):
    """Verify all atlas-sdk fields are preserved during JSONL loading."""
    payload = {
        "task": "Test task",
        "final_answer": "Test answer",
        "plan": {"steps": [{"id": 1, "description": "step 1"}]},
        "session_metadata": {
            "learning_key": "task-1",
            "teacher_notes": ["note1", "note2"],
            "reward_summary": {"avg": 0.8},
        },
        # Essential session fields from atlas-sdk
        "session_reward": {"score": 0.9, "rationale": "excellent"},
        "trajectory_events": [{"event": "step_completed", "step_id": 1}],
        "student_learning": "student cue text",
        "teacher_learning": "teacher guidance text",
        "learning_history": {"previous_attempts": 2, "improvement": 0.15},
        "adaptive_summary": {"mode": "coach", "intensity": "medium"},
        "steps": [
            {
                "step_id": 1,
                "description": "Execute step",
                "trace": "HUMAN: execute",
                "output": "result",
                "evaluation": {
                    "score": 0.9,
                    "rationale": "Good",
                    "judges": [],
                },
                "validation": {"valid": True},
                "attempts": 1,
                "guidance": [],
                # Essential step fields from atlas-sdk
                "runtime": {"duration_ms": 1500, "tokens": 250},
                "depends_on": [0],
                "artifacts": {"file": "output.txt", "size": 1024},
                "deliverable": "final output",
                "metadata": {
                    "attempt_history": [
                        {"attempt": 1, "score": 0.9}
                    ]
                },
            }
        ],
    }
    path = tmp_path / "sdk_export.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    sessions = load_runtime_traces(path)
    assert len(sessions) == 1
    session = sessions[0]

    # Verify essential session fields preserved
    assert session.task == "Test task"
    assert session.session_reward is not None
    assert session.session_reward["score"] == 0.9
    assert session.trajectory_events is not None
    assert len(session.trajectory_events) == 1
    assert session.student_learning == "student cue text"
    assert session.teacher_learning == "teacher guidance text"
    assert session.learning_history is not None
    assert session.learning_history["improvement"] == 0.15
    assert session.adaptive_summary is not None
    assert session.adaptive_summary["mode"] == "coach"

    # Verify property accessors work
    assert session.learning_key == "task-1"
    assert session.teacher_notes == ["note1", "note2"]
    assert session.reward_summary == {"avg": 0.8}

    # Verify essential step fields preserved
    step = session.steps[0]
    assert step.runtime is not None
    assert step.runtime["duration_ms"] == 1500
    assert step.depends_on == [0]
    assert step.artifacts is not None
    assert step.artifacts["file"] == "output.txt"
    assert step.deliverable == "final output"

    # Verify step property accessor works
    assert step.attempt_history is not None
    assert len(step.attempt_history) == 1

    # Verify to_dict() round-trip
    session_dict = session.to_dict()
    assert session_dict["session_reward"]["score"] == 0.9
    assert session_dict["student_learning"] == "student cue text"
    assert session_dict["steps"][0]["runtime"]["duration_ms"] == 1500


def test_backward_compatibility_missing_fields(tmp_path: Path):
    """Verify old JSONL exports without new fields still load correctly."""
    # Old SDK export format (no session_reward, learning fields, etc.)
    payload = {
        "task": "Old task",
        "final_answer": "Old answer",
        "plan": {"steps": [{"id": 1, "description": "step"}]},
        "session_metadata": {},
        "steps": [
            {
                "step_id": 1,
                "description": "step",
                "trace": "trace",
                "output": "output",
                "evaluation": {"score": 0.8, "judges": []},
            }
        ],
    }
    path = tmp_path / "old_export.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    sessions = load_runtime_traces(path)
    assert len(sessions) == 1
    session = sessions[0]

    # Old fields still work
    assert session.task == "Old task"
    assert session.final_answer == "Old answer"

    # New fields are None (graceful handling)
    assert session.session_reward is None
    assert session.trajectory_events is None
    assert session.student_learning is None
    assert session.teacher_learning is None
    step = session.steps[0]
    assert step.runtime is None
    assert step.depends_on is None


def test_drift_alert_fallback_logic(tmp_path: Path):
    """Verify drift_alert property checks both nested and top-level locations."""
    # Test case 1: drift_alert nested inside drift dict (SDK format)
    payload_nested = {
        "task": "Test task",
        "final_answer": "Test answer",
        "plan": {"steps": []},
        "session_metadata": {
            "drift": {
                "drift_alert": "nested_alert_value",
                "score": 0.5,
            }
        },
        "steps": [],
    }

    # Test case 2: drift_alert at top-level (legacy format)
    payload_top_level = {
        "task": "Test task 2",
        "final_answer": "Test answer 2",
        "plan": {"steps": []},
        "session_metadata": {
            "drift_alert": "top_level_alert_value",
        },
        "steps": [],
    }

    # Test case 3: drift_alert in both locations (nested takes precedence)
    payload_both = {
        "task": "Test task 3",
        "final_answer": "Test answer 3",
        "plan": {"steps": []},
        "session_metadata": {
            "drift": {
                "drift_alert": "nested_priority",
            },
            "drift_alert": "top_level_fallback",
        },
        "steps": [],
    }

    path = tmp_path / "drift_test.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(payload_nested) + "\n")
        f.write(json.dumps(payload_top_level) + "\n")
        f.write(json.dumps(payload_both) + "\n")

    sessions = load_runtime_traces(path)
    assert len(sessions) == 3

    # Verify nested location works
    assert sessions[0].drift_alert == "nested_alert_value"

    # Verify top-level fallback works
    assert sessions[1].drift_alert == "top_level_alert_value"

    # Verify nested takes precedence when both exist
    assert sessions[2].drift_alert == "nested_priority"


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
