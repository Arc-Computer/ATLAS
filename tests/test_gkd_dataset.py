"""Unit tests for GKD dataset pipeline."""

import pytest

pytest.importorskip("atlas.training_data")

from atlas_core.runtime import AtlasRewardBreakdown, AtlasSessionTrace, AtlasStepTrace

from trainers import gkd_dataset


def _make_session(
    *,
    session_id: int = 123,
    reward: float = 0.92,
    trajectory_events=None,
    learning_key: str = "test-task",
) -> AtlasSessionTrace:
    """Helper to create a mock AtlasSessionTrace."""
    step = AtlasStepTrace(
        step_id=1,
        description="Execute action",
        trace="Student attempted the step.",
        output="Student output text.",
        reward=AtlasRewardBreakdown(score=reward),
        guidance=["Teacher guidance"],
    )
    return AtlasSessionTrace(
        task="Complete the workflow",
        final_answer="Final answer.",
        plan={"steps": [{"id": 1, "description": "Execute action"}]},
        steps=[step],
        session_metadata={
            "session_id": session_id,
            "learning_key": learning_key,
            "status": "succeeded",
        },
        session_reward={"score": reward},
        trajectory_events=trajectory_events or [],
    )


def test_load_gkd_conversations_basic(monkeypatch):
    """Test loading conversations from Postgres."""
    events = [
        {
            "actor": "teacher",
            "event": {"payload": {"message": "Teacher guidance here"}},
        },
        {
            "actor": "student",
            "event": {"payload": {"message": "Student attempt here"}},
        },
    ]
    session = _make_session(trajectory_events=events, reward=0.85)

    def mock_stream_conversations(*args, **kwargs):
        return [
            {
                "messages": [
                    {"role": "assistant", "content": "Teacher guidance here"},
                    {"role": "user", "content": "Student attempt here"},
                ],
                "session_id": 123,
                "learning_key": "test-task",
                "reward": 0.85,
            }
        ]

    monkeypatch.setattr(
        gkd_dataset,
        "stream_conversations_from_postgres",
        mock_stream_conversations,
    )

    conversations = gkd_dataset.load_gkd_conversations(
        db_url="postgresql://stub",
        min_reward=0.8,
    )

    assert len(conversations) == 1
    assert conversations[0]["session_id"] == 123
    assert conversations[0]["reward"] == 0.85
    assert len(conversations[0]["messages"]) == 2


def test_load_gkd_conversations_filters_by_reward(monkeypatch):
    """Test that conversations below min_reward are filtered."""

    def mock_stream_conversations(*args, **kwargs):
        min_reward = kwargs.get("min_reward", 0.0)
        # Only return sessions above threshold
        if min_reward <= 0.85:
            return [{"messages": [], "session_id": 1, "reward": 0.85}]
        return []

    monkeypatch.setattr(
        gkd_dataset,
        "stream_conversations_from_postgres",
        mock_stream_conversations,
    )

    # Should get conversation with reward 0.85
    convs_low = gkd_dataset.load_gkd_conversations(
        db_url="postgresql://stub",
        min_reward=0.8,
    )
    assert len(convs_low) == 1

    # Should not get conversation (filtered out)
    convs_high = gkd_dataset.load_gkd_conversations(
        db_url="postgresql://stub",
        min_reward=0.9,
    )
    assert len(convs_high) == 0


def test_load_gkd_conversations_filters_by_learning_key(monkeypatch):
    """Test filtering by learning key."""

    def mock_stream_conversations(*args, **kwargs):
        learning_key = kwargs.get("learning_key")
        if learning_key == "crm_workflows":
            return [
                {
                    "messages": [],
                    "session_id": 1,
                    "learning_key": "crm_workflows",
                    "reward": 0.9,
                }
            ]
        elif learning_key is None:
            return [
                {
                    "messages": [],
                    "session_id": 1,
                    "learning_key": "crm_workflows",
                    "reward": 0.9,
                },
                {"messages": [], "session_id": 2, "learning_key": "other_task", "reward": 0.85},
            ]
        return []

    monkeypatch.setattr(
        gkd_dataset,
        "stream_conversations_from_postgres",
        mock_stream_conversations,
    )

    # Filter by specific learning key
    convs_filtered = gkd_dataset.load_gkd_conversations(
        db_url="postgresql://stub",
        learning_key="crm_workflows",
    )
    assert len(convs_filtered) == 1
    assert convs_filtered[0]["learning_key"] == "crm_workflows"

    # No filter (all learning keys)
    convs_all = gkd_dataset.load_gkd_conversations(
        db_url="postgresql://stub",
        learning_key=None,
    )
    assert len(convs_all) == 2


def test_build_gkd_dataset_creates_train_eval_split(monkeypatch):
    """Test that datasets are properly split into train/eval."""
    mock_conversations = [
        {"messages": [], "session_id": i, "reward": 0.9} for i in range(100)
    ]

    def mock_load_conversations(*args, **kwargs):
        return mock_conversations

    monkeypatch.setattr(
        gkd_dataset,
        "load_gkd_conversations",
        mock_load_conversations,
    )

    train_ds, eval_ds = gkd_dataset.build_gkd_dataset(
        db_url="postgresql://stub",
        eval_split=0.2,
        shuffle=False,
        seed=42,
    )

    # Check split ratio (approximately 80/20)
    assert len(train_ds) == 80
    assert len(eval_ds) == 20

    # Check dataset format
    assert "messages" in train_ds.column_names
    assert "session_id" in train_ds.column_names
    assert "reward" in train_ds.column_names


def test_build_gkd_dataset_no_eval_split(monkeypatch):
    """Test building dataset without eval split."""
    mock_conversations = [
        {"messages": [], "session_id": i, "reward": 0.9} for i in range(50)
    ]

    def mock_load_conversations(*args, **kwargs):
        return mock_conversations

    monkeypatch.setattr(
        gkd_dataset,
        "load_gkd_conversations",
        mock_load_conversations,
    )

    train_ds, eval_ds = gkd_dataset.build_gkd_dataset(
        db_url="postgresql://stub",
        eval_split=0.0,
    )

    # All data in train, eval should be empty
    assert len(train_ds) == 50
    assert len(eval_ds) == 0


def test_build_gkd_dataset_raises_on_no_data(monkeypatch):
    """Test that ValueError is raised when no conversations available."""

    def mock_load_conversations(*args, **kwargs):
        return []

    monkeypatch.setattr(
        gkd_dataset,
        "load_gkd_conversations",
        mock_load_conversations,
    )

    with pytest.raises(ValueError, match="No conversations available"):
        gkd_dataset.build_gkd_dataset(db_url="postgresql://stub")


def test_get_gkd_postgres_dataset_returns_dict(monkeypatch):
    """Test Hydra-compatible function returns dict format."""
    mock_conversations = [
        {"messages": [], "session_id": i, "reward": 0.9} for i in range(10)
    ]

    def mock_load_conversations(*args, **kwargs):
        return mock_conversations

    monkeypatch.setattr(
        gkd_dataset,
        "load_gkd_conversations",
        mock_load_conversations,
    )

    result = gkd_dataset.get_gkd_postgres_dataset(
        db_url="postgresql://stub",
        eval_split=0.2,
    )

    # Check dict format with expected keys
    assert isinstance(result, dict)
    assert "train_dataset" in result
    assert "eval_dataset" in result
    assert len(result["train_dataset"]) == 8
    assert len(result["eval_dataset"]) == 2


def test_load_gkd_conversations_respects_limit(monkeypatch):
    """Test that limit parameter is respected."""

    def mock_stream_conversations(*args, **kwargs):
        limit = kwargs.get("limit")
        if limit is None:
            return [{"messages": [], "session_id": i, "reward": 0.9} for i in range(100)]
        return [{"messages": [], "session_id": i, "reward": 0.9} for i in range(limit)]

    monkeypatch.setattr(
        gkd_dataset,
        "stream_conversations_from_postgres",
        mock_stream_conversations,
    )

    # With limit
    convs_limited = gkd_dataset.load_gkd_conversations(
        db_url="postgresql://stub",
        limit=10,
    )
    assert len(convs_limited) == 10

    # Without limit
    convs_unlimited = gkd_dataset.load_gkd_conversations(
        db_url="postgresql://stub",
        limit=None,
    )
    assert len(convs_unlimited) == 100


def test_conversation_format_is_trl_compatible(monkeypatch):
    """Test that conversation format matches TRL expectations."""
    mock_conversation = {
        "messages": [
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant message"},
        ],
        "session_id": 123,
        "reward": 0.95,
        "learning_key": "test-task",
    }

    def mock_load_conversations(*args, **kwargs):
        return [mock_conversation]

    monkeypatch.setattr(
        gkd_dataset,
        "load_gkd_conversations",
        mock_load_conversations,
    )

    train_ds, _ = gkd_dataset.build_gkd_dataset(
        db_url="postgresql://stub",
        eval_split=0.0,
    )

    # Check TRL format requirements
    record = train_ds[0]
    assert "messages" in record
    assert isinstance(record["messages"], list)
    assert all("role" in msg and "content" in msg for msg in record["messages"])
    assert all(msg["role"] in ["user", "assistant", "system"] for msg in record["messages"])
