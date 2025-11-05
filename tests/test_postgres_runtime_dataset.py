import pytest

pytest.importorskip("atlas.training_data")

from atlas_core.runtime import AtlasRewardBreakdown, AtlasSessionTrace, AtlasStepTrace

from trainers import postgres_runtime_dataset as pg_dataset


def _make_session(
    *,
    trajectory_events=None,
    guidance=None,
    step_output: str = "Student output text.",
) -> AtlasSessionTrace:
    step = AtlasStepTrace(
        step_id=1,
        description="Execute action",
        trace="Student attempted the step.",
        output=step_output,
        reward=AtlasRewardBreakdown(score=0.9),
        guidance=list(guidance or []),
    )
    return AtlasSessionTrace(
        task="Complete the assigned workflow",
        final_answer="Final student answer.",
        plan={"steps": [{"id": 1, "description": "Execute action"}]},
        steps=[step],
        session_metadata={"session_id": 123, "learning_key": "demo-learning"},
        session_reward={"score": 0.92},
        trajectory_events=trajectory_events,
    )


def test_build_conversation_dataset_converts_events(monkeypatch):
    events = [
        {
            "actor": "teacher",
            "event": {
                "event_type": "GUIDANCE",
                "metadata": {"actor": "teacher"},
                "payload": {"message": "Keep answers concise."},
            },
        },
        {
            "actor": "student",
            "event": {
                "event_type": "ATTEMPT",
                "metadata": {"actor": "student"},
                "payload": {"message": "Student attempted solution."},
            },
        },
    ]

    session = _make_session(trajectory_events=events, guidance=["Fallback guidance"])

    monkeypatch.setattr(
        pg_dataset,
        "get_training_sessions",
        lambda *args, **kwargs: [session],
    )

    result = pg_dataset.build_conversation_dataset(
        "postgresql://stub",
        eval_split_ratio=0.0,
        shuffle=False,
    )

    dataset = result["train_dataset"]
    assert dataset is not None
    assert len(dataset) == 1
    record = dataset[0]
    roles = [message["role"] for message in record["messages"]]
    assert roles == ["system", "assistant", "user"]
    contents = [message["content"] for message in record["messages"][1:]]
    assert contents == ["Keep answers concise.", "Student attempted solution."]
    assert record["session_id"] == 123
    assert record["learning_key"] == "demo-learning"
    assert pytest.approx(record["reward"], rel=1e-6) == 0.92


def test_build_conversation_dataset_falls_back_to_step_guidance(monkeypatch):
    session = _make_session(trajectory_events=None, guidance=["Provide evidence."])

    monkeypatch.setattr(
        pg_dataset,
        "get_training_sessions",
        lambda *args, **kwargs: [session],
    )

    result = pg_dataset.build_conversation_dataset(
        "postgresql://stub",
        eval_split_ratio=0.0,
        shuffle=False,
    )

    record = result["train_dataset"][0]
    assert [m["role"] for m in record["messages"]] == ["system", "assistant", "user"]
    assert record["messages"][1]["content"] == "Provide evidence."
    assert record["messages"][2]["content"] == "Student output text."


def test_build_conversation_dataset_requires_db_url(monkeypatch):
    monkeypatch.setattr(
        pg_dataset,
        "get_training_sessions",
        lambda *args, **kwargs: [],
    )

    with pytest.raises(ValueError):
        pg_dataset.build_conversation_dataset("", eval_split_ratio=0.0)
