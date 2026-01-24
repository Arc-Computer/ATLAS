"""Shared dataclasses representing runtime reward and trace payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class AtlasJudgeSample:
    """Fine-grained sample emitted by a reward judge."""

    score: float
    rationale: str
    principles: List[Dict[str, Any]] = field(default_factory=list)
    uncertainty: Optional[float] = None
    temperature: Optional[float] = None


@dataclass
class AtlasJudgeBreakdown:
    """Structured result from a single reward judge."""

    identifier: str
    score: float
    rationale: str
    principles: List[Dict[str, Any]] = field(default_factory=list)
    samples: List[AtlasJudgeSample] = field(default_factory=list)
    escalated: bool = False
    escalation_reason: Optional[str] = None


@dataclass
class AtlasRewardBreakdown:
    """Aggregated reward summary for a step or episode."""

    score: float
    judges: List[AtlasJudgeBreakdown] = field(default_factory=list)
    rationale: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly dict."""

        return {
            "score": self.score,
            "rationale": self.rationale,
            "judges": [
                {
                    "identifier": judge.identifier,
                    "score": judge.score,
                    "rationale": judge.rationale,
                    "principles": judge.principles,
                    "samples": [
                        {
                            "score": sample.score,
                            "rationale": sample.rationale,
                            "principles": sample.principles,
                            "uncertainty": sample.uncertainty,
                            "temperature": sample.temperature,
                        }
                        for sample in judge.samples
                    ],
                    "escalated": judge.escalated,
                    "escalation_reason": judge.escalation_reason,
                }
                for judge in self.judges
            ],
            "raw": self.raw,
        }


@dataclass
class AtlasStepTrace:
    """Single plan step with execution, validation, and reward context."""

    step_id: int
    description: str
    trace: str
    output: str
    reward: AtlasRewardBreakdown
    tool: Optional[str] = None
    tool_params: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    attempts: int = 1
    guidance: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Essential step fields (match atlas-sdk PR #121)
    runtime: Optional[Dict[str, Any]] = None  # Execution telemetry
    depends_on: Optional[List[Union[int, str]]] = None  # Step dependencies
    artifacts: Optional[Dict[str, Any]] = None  # Step artifacts
    deliverable: Optional[str] = None  # Step deliverable

    @property
    def attempt_history(self) -> Optional[List[Dict[str, Any]]]:
        """Property accessor for attempt_history from metadata."""
        return self.metadata.get("attempt_history")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-friendly dict with all fields preserved."""
        return {
            "step_id": self.step_id,
            "description": self.description,
            "trace": self.trace,
            "output": self.output,
            "reward": self.reward.to_dict() if isinstance(self.reward, AtlasRewardBreakdown) else self.reward,
            "tool": self.tool,
            "tool_params": self.tool_params,
            "context": self.context,
            "validation": self.validation,
            "attempts": self.attempts,
            "guidance": self.guidance,
            "metadata": self.metadata,
            # Essential step fields
            "runtime": self.runtime,
            "depends_on": self.depends_on,
            "artifacts": self.artifacts,
            "deliverable": self.deliverable,
        }


@dataclass
class AtlasSessionTrace:
    """Complete session exported from the runtime."""

    task: str
    final_answer: str
    plan: Dict[str, Any]
    steps: List[AtlasStepTrace]
    session_metadata: Dict[str, Any] = field(default_factory=dict)
    # Essential training fields (match atlas-sdk PR #121)
    session_reward: Optional[Dict[str, Any]] = None  # GRPO training
    trajectory_events: Optional[List[Dict[str, Any]]] = None  # All workflows
    student_learning: Optional[str] = None  # Distillation
    teacher_learning: Optional[str] = None  # Distillation
    learning_history: Optional[Dict[str, Any]] = None  # Transfer learning
    adaptive_summary: Optional[Dict[str, Any]] = None  # Execution context

    # Property accessors for optional fields (already in session_metadata)
    @property
    def learning_key(self) -> Optional[str]:
        """Property accessor for learning_key from session_metadata."""
        return self.session_metadata.get("learning_key")

    @property
    def teacher_notes(self) -> Optional[List[Any]]:
        """Property accessor for teacher_notes from session_metadata."""
        return self.session_metadata.get("teacher_notes")

    @property
    def reward_summary(self) -> Optional[Dict[str, Any]]:
        """Property accessor for reward_summary from session_metadata."""
        return self.session_metadata.get("reward_summary")

    @property
    def drift(self) -> Optional[Dict[str, Any]]:
        """Property accessor for drift from session_metadata."""
        return self.session_metadata.get("drift")

    @property
    def drift_alert(self) -> Optional[Any]:
        """Get drift alert flag from session metadata.

        Checks both drift.drift_alert (nested) and session_metadata.drift_alert
        (top-level) for backward compatibility with different export formats.
        """
        drift_payload = self.session_metadata.get("drift")
        if drift_payload is None:
            return self.session_metadata.get("drift_alert")
        if isinstance(drift_payload, dict):
            return drift_payload.get("drift_alert") or self.session_metadata.get("drift_alert")
        return self.session_metadata.get("drift_alert")

    @property
    def triage_dossier(self) -> Optional[Dict[str, Any]]:
        """Property accessor for triage_dossier from session_metadata."""
        return self.session_metadata.get("triage_dossier")

    @property
    def reward_audit(self) -> Optional[List[Dict[str, Any]]]:
        """Property accessor for reward_audit from session_metadata."""
        return self.session_metadata.get("reward_audit")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-friendly dict with all fields preserved."""
        return {
            "task": self.task,
            "final_answer": self.final_answer,
            "plan": self.plan,
            "steps": [step.to_dict() for step in self.steps],
            "session_metadata": self.session_metadata,
            # Essential training fields
            "session_reward": self.session_reward,
            "trajectory_events": self.trajectory_events,
            "student_learning": self.student_learning,
            "teacher_learning": self.teacher_learning,
            "learning_history": self.learning_history,
            "adaptive_summary": self.adaptive_summary,
        }
