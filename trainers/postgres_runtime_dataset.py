"""Dataset helper for streaming Atlas runtime sessions directly from Postgres."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from datasets import Dataset

try:
    from atlas.training_data import get_training_sessions
except ImportError as exc:  # pragma: no cover - surfaced during runtime, not tests
    raise ImportError(
        "atlas-sdk is required for Postgres-backed dataset loading. "
        "Install atlas-sdk>=0.1.14 to use trainers.postgres_runtime_dataset."
    ) from exc

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from atlas.runtime.schema import AtlasSessionTrace

logger = logging.getLogger(__name__)

_TEXT_KEYS: Tuple[str, ...] = (
    "message",
    "text",
    "content",
    "guidance",
    "explanation",
    "output",
    "body",
    "value",
    "utterance",
    "prompt",
)


def _normalise_actor(actor: Optional[str]) -> Optional[str]:
    if not actor:
        return None
    normalised = actor.strip().lower()
    if not normalised:
        return None
    return normalised


def _actor_to_role(actor: Optional[str]) -> Optional[str]:
    actor_norm = _normalise_actor(actor)
    if actor_norm is None:
        return None
    if actor_norm in {"student", "user", "executor", "runtime_student"}:
        return "user"
    if actor_norm in {"teacher", "coach", "atlas_teacher", "assistant"}:
        return "assistant"
    if actor_norm in {"system", "runtime"}:
        return "system"
    return None


def _extract_from_openai_content(payload: Any) -> Optional[str]:
    if not isinstance(payload, list):
        return None
    parts: List[str] = []
    for item in payload:
        if isinstance(item, dict):
            # OpenAI style: {"type": "text", "text": "..."}
            text_value = item.get("text")
            if isinstance(text_value, str) and text_value.strip():
                parts.append(text_value.strip())
    if parts:
        return "\n".join(parts)
    return None


def _search_for_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            text = _search_for_text(item)
            if text:
                parts.append(text)
        if parts:
            return "\n".join(parts)
        return None
    if isinstance(value, dict):
        # OpenAI chat format support
        if "content" in value and isinstance(value["content"], list):
            openai_text = _extract_from_openai_content(value["content"])
            if openai_text:
                return openai_text
        if "message" in value and isinstance(value["message"], dict):
            nested = _search_for_text(value["message"])
            if nested:
                return nested
        for key in _TEXT_KEYS:
            if key in value:
                text = _search_for_text(value[key])
                if text:
                    return text
        # Some payloads embed messages beneath 'payload', 'data', or 'metadata'.
        for additional_key in ("payload", "data", "metadata", "details"):
            if additional_key in value:
                text = _search_for_text(value[additional_key])
                if text:
                    return text
    return None


def _extract_event_actor(event: Dict[str, Any]) -> Optional[str]:
    actor = event.get("actor")
    if actor:
        return actor
    event_payload = event.get("event")
    if isinstance(event_payload, dict):
        metadata = event_payload.get("metadata")
        if isinstance(metadata, dict):
            actor = metadata.get("actor")
            if actor:
                return actor
        actor = event_payload.get("actor")
        if actor:
            return actor
    metadata = event.get("metadata")
    if isinstance(metadata, dict):
        actor = metadata.get("actor")
        if actor:
            return actor
    return None


def _extract_event_text(event: Dict[str, Any]) -> Optional[str]:
    for key in ("message", "text", "guidance", "content"):
        value = event.get(key)
        text = _search_for_text(value)
        if text:
            return text
    event_payload = event.get("event")
    if isinstance(event_payload, dict):
        text = _search_for_text(event_payload)
        if text:
            return text
    return None


def _messages_from_events(events: Sequence[dict[str, Any]]) -> List[dict[str, str]]:
    messages: List[dict[str, str]] = []
    for entry in events:
        if not isinstance(entry, dict):
            continue
        role = _actor_to_role(_extract_event_actor(entry))
        if role is None:
            continue
        text = _extract_event_text(entry)
        if text is None:
            continue
        messages.append({"role": role, "content": text})
    return messages


def _messages_from_steps(session: "AtlasSessionTrace") -> List[dict[str, str]]:
    fallback_messages: List[dict[str, str]] = []
    for step in getattr(session, "steps", []) or []:
        guidance = getattr(step, "guidance", None)
        if isinstance(guidance, list):
            for note in guidance:
                if isinstance(note, str) and note.strip():
                    fallback_messages.append({"role": "assistant", "content": note.strip()})
        output = getattr(step, "output", None)
        if isinstance(output, str) and output.strip():
            fallback_messages.append({"role": "user", "content": output.strip()})
        elif isinstance(getattr(step, "trace", None), str) and step.trace.strip():
            fallback_messages.append({"role": "user", "content": step.trace.strip()})
    return fallback_messages


def session_to_conversation(
    session: "AtlasSessionTrace",
    *,
    include_system_message: bool = True,
    drop_empty: bool = True,
) -> Optional[dict[str, Any]]:
    """Convert an Atlas session trace to a TRL-friendly conversation record."""
    if session is None:
        return None

    messages: List[dict[str, str]] = []

    if include_system_message:
        plan = getattr(session, "plan", {}) or {}
        plan_steps = plan.get("steps") if isinstance(plan, dict) else None
        plan_summary: List[str] = []
        if isinstance(plan_steps, list):
            for step in plan_steps:
                if isinstance(step, dict):
                    description = step.get("description")
                    if description:
                        identifier = step.get("id")
                        if identifier is not None:
                            plan_summary.append(f"{identifier}: {description}")
                        else:
                            plan_summary.append(str(description))
        context_lines = [f"Task: {(getattr(session, 'task', None) or '').strip()}"]
        if plan_summary:
            context_lines.append("Plan:")
            context_lines.extend(f"- {line}" for line in plan_summary)
        system_content = "\n".join(line for line in context_lines if line)
        if system_content:
            messages.append({"role": "system", "content": system_content})

    events = getattr(session, "trajectory_events", None) or []
    event_messages = _messages_from_events(events)

    if not event_messages:
        fallback_messages = _messages_from_steps(session)
        event_messages = fallback_messages

    if drop_empty and not event_messages:
        return None

    messages.extend(event_messages)

    if drop_empty and len(messages) <= (1 if include_system_message else 0):
        return None

    prompt_messages = messages[:-1] if len(messages) > 1 else messages[:]
    completion_message = messages[-1] if messages else None

    def _serialize_messages(items: List[dict[str, str]]) -> str:
        serialized: List[str] = []
        for item in items:
            role = item.get("role", "").strip()
            content = item.get("content", "").strip()
            if not content:
                continue
            serialized.append(f"{role}: {content}" if role else content)
        return "\n".join(serialized)

    session_metadata = getattr(session, "session_metadata", {}) or {}
    session_id = session_metadata.get("session_id") or session_metadata.get("id")
    if session_id is None:
        session_id = getattr(session, "session_id", None)
    reward = None
    session_reward = getattr(session, "session_reward", None)
    if isinstance(session_reward, dict):
        reward = session_reward.get("score")

    record: dict[str, Any] = {
        "messages": messages,
        "session_id": session_id,
        "learning_key": session_metadata.get("learning_key"),
        "reward": reward,
        "prompt_text": _serialize_messages(prompt_messages),
        "completion_text": completion_message.get("content", "").strip() if completion_message else "",
    }

    # Preserve lightweight metadata for filtering without embedding raw session dict.
    for key in ("execution_mode", "status", "created_at", "completed_at"):
        if key in session_metadata:
            record.setdefault("metadata", {})[key] = session_metadata[key]
    if getattr(session, "student_learning", None):
        record.setdefault("metadata", {})["student_learning"] = session.student_learning
    if getattr(session, "teacher_learning", None):
        record.setdefault("metadata", {})["teacher_learning"] = session.teacher_learning

    return record


def stream_conversations_from_postgres(
    db_url: str,
    *,
    min_reward: float = 0.0,
    limit: Optional[int] = None,
    offset: int = 0,
    learning_key: Optional[str] = None,
    status_filters: Optional[Sequence[str]] = None,
    review_status_filters: Optional[Sequence[str]] = None,
    include_trajectory_events: bool = True,
    include_learning_data: bool = True,
) -> List[dict[str, Any]]:
    """Load sessions from Postgres and convert them into conversation records."""

    # get_training_sessions applies a default limit of 1000; preserve behaviour unless overridden.
    sessions = get_training_sessions(
        db_url,
        min_reward=min_reward,
        limit=limit if limit is not None else 1000,
        offset=offset,
        learning_key=learning_key,
        status_filters=status_filters,
        review_status_filters=review_status_filters,
        include_trajectory_events=include_trajectory_events,
        include_learning_data=include_learning_data,
    )

    records: List[dict[str, Any]] = []
    dropped = 0
    for session in sessions:
        record = session_to_conversation(session)
        if record is None:
            dropped += 1
            continue
        records.append(record)

    if dropped:
        logger.debug("Dropped %s sessions without usable messages", dropped)

    return records


def build_conversation_dataset(
    db_url: str,
    *,
    min_reward: float = 0.0,
    limit: Optional[int] = None,
    offset: int = 0,
    learning_key: Optional[str] = None,
    status_filters: Optional[Sequence[str]] = None,
    review_status_filters: Optional[Sequence[str]] = None,
    eval_split_ratio: float = 0.1,
    shuffle: bool = True,
    dataset_max_sessions: Optional[int] = None,
    include_trajectory_events: bool = True,
    include_learning_data: bool = True,
    seed: int = 42,
) -> Dict[str, Optional[Dataset]]:
    """Return train/eval HuggingFace datasets backed by Postgres conversations."""

    db_url = db_url or os.environ.get("ATLAS_DATABASE_URL") or ""
    if not db_url:
        raise ValueError("Postgres dataset loader requires `db_url` or ATLAS_DATABASE_URL to be set.")

    records = stream_conversations_from_postgres(
        db_url,
        min_reward=min_reward,
        limit=limit,
        offset=offset,
        learning_key=learning_key,
        status_filters=status_filters,
        review_status_filters=review_status_filters,
        include_trajectory_events=include_trajectory_events,
        include_learning_data=include_learning_data,
    )

    if dataset_max_sessions is not None:
        records = records[: dataset_max_sessions]

    if not records:
        raise ValueError("No conversations available for the specified filters.")

    dataset = Dataset.from_list(records)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    if eval_split_ratio and 0 < eval_split_ratio < 1:
        dataset_dict = dataset.train_test_split(test_size=eval_split_ratio, seed=seed)
        return {"train_dataset": dataset_dict["train"], "eval_dataset": dataset_dict["test"]}

    return {"train_dataset": dataset, "eval_dataset": None}
