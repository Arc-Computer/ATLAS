"""Shared dataclasses representing runtime reward and trace payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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


@dataclass
class AtlasSessionTrace:
    """Complete session exported from the runtime."""

    task: str
    final_answer: str
    plan: Dict[str, Any]
    steps: List[AtlasStepTrace]
    session_metadata: Dict[str, Any] = field(default_factory=dict)
