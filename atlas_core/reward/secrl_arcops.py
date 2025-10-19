"""SecRL ArcOps-Cyber reward integration for the Atlas runtime."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from atlas.config.models import AtlasConfig, LLMParameters, RewardObjectiveConfig
from atlas.evaluation.evaluator import RewardEvaluation, SessionTrajectory
from atlas.runtime.schema import AtlasRewardBreakdown
from atlas.utils.llm_client import LLMClient


EVAL_ANSWER_TEMPLATE = """Question: {question}

Golden Answer: {golden_answer}

Submitted Answer: {submitted_answer}"""

EVAL_SOLUTION_TEMPLATE = """Question: {question}

Golden Solution:
{golden_solution}

Submitted Answer:
{submitted_answer}"""

FUZZY_ANSWER_CHECK_PROMPT = """Given a golden answer to a security question and a submitted answer, please evaluate whether the submitted answer matches the golden answer.
You are given:
- The question
- The golden answer
- The submitted answer

Note:
The submitted answer does not need to match the golden answer exactly. But the key content should be present.
If the submitted answer presents the golden answer along with additional context, it should be considered correct.
If the submitted answer is an overly large enumeration (>15 is the strict limit) that includes the golden answer and lacks relevance, it should be considered false. All enumerations less than 10 and containing the golden answer should be considered correct. Between 10 and 15, use your discretion to determine if the answer is relevant enough to be considered correct.
If the format of the submitted answer is different from the golden answer but the meaning is the same, it should be considered as true. Ignore the case of the text.
For time-based questions, the submitted answer should be within a reasonable time frame of the golden answer and the format of the timestamps is not required to match exactly.
For domain-specific questions, the submitted answer should contain the key information mentioned in the golden answer. Ignore differences in http/https, www, and trailing slashes in URLs.
In case you find discrepancies between the question and the golden answer, please consider the golden answer as the ground truth as you do not have full context of the question.

First give a brief analysis using 1-2 short sentences, then give your decision.
Follow this format:
Analysis: <your analysis>
Is_Answer_Correct: <"True" or "False">"""

STRICT_ANSWER_CHECK_PROMPT = """Given a golden answer to a security question and a submitted answer, please evaluate whether the submitted answer matches the golden answer.
You are given:
- The question
- The golden answer
- The submitted answer

Note:
The submitted answer must match the golden answer exactly. Literal equality is required.

First give a brief analysis using 1-2 short sentences, then give your decision.
Follow this format:
Analysis: <your analysis>
Is_Answer_Correct: <"True" or "False">"""

STEP_CHECK_PROMPT = """Given a security question, a submitted answer, and a ground truth solution, please evaluate the correctness of the submitted answer.
The ground truth solution may contain several steps. Please go through each step of the ground truth solution and evaluate whether the given answer correctly contains key info (the Indicator of Compromise) of that step, which is usually enclosed in `< >`.
Note:
- If the format of the submitted answer is different from the golden answer but the meaning is the same, it should be considered as true.
- The key information should not be the ones that is already present in the question.

Your response must be valid JSON:
{
    "<step_i>": {
        "analysis": "<your analysis>",
        "is_step_correct": "<True or False>"
    },
    ...
}
`step_i` indexes must start from 0."""


@dataclass
class _AnswerEvaluation:
    reward: float
    response: str | None = None
    decision: str | None = None
    error: str | None = None


@dataclass
class _SolutionEvaluation:
    reward: float
    response: str | None = None
    steps: List[Dict[str, Any]] | None = None
    error: str | None = None


class SecRLArcOpsReward:
    """Implements the Microsoft SecRL evaluation protocol for ArcOps-Cyber."""

    def __init__(
        self,
        answer_llm: LLMParameters,
        *,
        step_llm: LLMParameters | None = None,
        strict: bool = False,
        step_check: bool = True,
        discount_factor: float = 0.4,
    ) -> None:
        self._answer_client = LLMClient(answer_llm)
        self._step_client = LLMClient(step_llm or answer_llm)
        self._answer_prompt = STRICT_ANSWER_CHECK_PROMPT if strict else FUZZY_ANSWER_CHECK_PROMPT
        self._step_check = step_check
        self._discount_factor = float(discount_factor)

    async def aevaluate_session(self, trajectory: SessionTrajectory) -> RewardEvaluation:
        metadata = trajectory.session_metadata or {}
        question_text = self._compose_full_question(metadata)
        golden_answer = metadata.get("gold_answer")
        solution_steps = metadata.get("solution_steps") or metadata.get("solution")

        if not golden_answer:
            breakdown = AtlasRewardBreakdown(score=0.0, rationale="SecRL reward unavailable: missing gold answer.")
            return RewardEvaluation(breakdown, student_learning="No SecRL reward computed.", teacher_learning=None)

        answer_eval = await self._evaluate_answer(question_text, golden_answer, trajectory.final_answer)
        score = float(answer_eval.reward)

        solution_eval: _SolutionEvaluation | None = None
        if score < 1.0 and self._step_check and solution_steps:
            solution_eval = await self._evaluate_solution(question_text, solution_steps, trajectory.final_answer)
            if solution_eval.reward:
                score = max(score, float(solution_eval.reward))

        raw_payload: Dict[str, Any] = {
            "answer": {
                "response": answer_eval.response,
                "decision": answer_eval.decision,
                "reward": answer_eval.reward,
                "error": answer_eval.error,
            }
        }
        if solution_eval is not None:
            raw_payload["solution"] = {
                "response": solution_eval.response,
                "reward": solution_eval.reward,
                "error": solution_eval.error,
                "steps": solution_eval.steps,
                "discount_factor": self._discount_factor,
            }

        rationale = self._build_rationale(answer_eval, solution_eval)
        breakdown = AtlasRewardBreakdown(score=score, rationale=rationale, raw={"secrl": raw_payload})
        student_learning = self._build_student_learning(score, answer_eval, solution_eval)
        return RewardEvaluation(breakdown, student_learning=student_learning, teacher_learning=None)

    async def _evaluate_answer(self, question: str, golden_answer: str, submitted_answer: str) -> _AnswerEvaluation:
        prompt = EVAL_ANSWER_TEMPLATE.format(
            question=question,
            golden_answer=golden_answer,
            submitted_answer=submitted_answer or "",
        )
        messages = [
            {"role": "system", "content": self._answer_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            response = await self._answer_client.acomplete(messages)
        except Exception as exc:  # pragma: no cover - defensive
            return _AnswerEvaluation(reward=0.0, error=str(exc))

        text = (response.content or "").strip()
        match = re.search(r"Is_Answer_Correct:\s*(True|False)", text, re.IGNORECASE)
        decision = match.group(1).lower() if match else None
        reward = 1.0 if decision == "true" else 0.0
        return _AnswerEvaluation(reward=reward, response=text, decision=decision)

    async def _evaluate_solution(self, question: str, solution_steps: Any, submitted_answer: str) -> _SolutionEvaluation:
        golden_solution = self._format_solution_steps(solution_steps)
        prompt = EVAL_SOLUTION_TEMPLATE.format(
            question=question,
            golden_solution=golden_solution,
            submitted_answer=submitted_answer or "",
        )
        messages = [
            {"role": "system", "content": STEP_CHECK_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            response = await self._step_client.acomplete(messages, response_format={"type": "json_object"})
        except Exception as exc:  # pragma: no cover - defensive
            return _SolutionEvaluation(reward=0.0, error=str(exc))

        payload = self._parse_solution_payload(response.content)
        if payload is None:
            return _SolutionEvaluation(reward=0.0, response=response.content)

        steps = self._normalise_steps(payload)
        reward = self._compute_step_reward(steps)
        return _SolutionEvaluation(reward=reward, response=response.content, steps=steps)

    def _parse_solution_payload(self, content: str) -> Dict[str, Any] | None:
        text = (content or "").strip()
        if not text:
            return None
        if text.startswith("```json"):
            text = text.split("```json", 1)[-1]
            text = text.split("```", 1)[0]
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _normalise_steps(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        items: List[Tuple[int, Dict[str, Any]]] = []
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            index = self._extract_step_index(key)
            value = dict(value)
            value["step"] = index
            items.append((index, value))
        items.sort(key=lambda pair: pair[0])
        return [entry for _, entry in items]

    def _compute_step_reward(self, steps: Iterable[Dict[str, Any]]) -> float:
        ordered = list(steps)
        if not ordered:
            return 0.0
        flags = [self._interpret_bool(item.get("is_step_correct")) for item in ordered]
        reversed_flags = list(reversed(flags))
        total = 0.0
        current = float(self._discount_factor)
        for flag in reversed_flags[1:]:
            if flag:
                total += current
                if total >= 1.0:
                    return 1.0
            current *= float(self._discount_factor)
        return min(total, 1.0)

    def _interpret_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "t", "yes", "y"}:
                return True
            if lowered in {"false", "f", "no", "n"}:
                return False
        return False

    def _extract_step_index(self, key: Any) -> int:
        if isinstance(key, str):
            match = re.search(r"(\d+)", key)
            if match:
                return int(match.group(1))
        try:
            return int(key)
        except (TypeError, ValueError):
            return 0

    def _format_solution_steps(self, steps: Any) -> str:
        if isinstance(steps, list):
            lines = [f"Step {idx}: {str(item)}" for idx, item in enumerate(steps)]
            return "\n".join(lines)
        return str(steps)

    def _compose_full_question(self, metadata: Dict[str, Any]) -> str:
        context = metadata.get("context")
        question = metadata.get("question") or metadata.get("task") or ""
        if context:
            return f"{context.strip()}\n\nQuestion: {question.strip()}"
        return question.strip()

    def _build_rationale(self, answer_eval: _AnswerEvaluation, solution_eval: _SolutionEvaluation | None) -> str:
        if answer_eval.reward >= 1.0:
            return "SecRL evaluation: submitted answer matched the golden answer."
        if solution_eval and solution_eval.reward > 0:
            matched = sum(1 for step in solution_eval.steps or [] if self._interpret_bool(step.get("is_step_correct")))
            total = len(solution_eval.steps or [])
            return f"SecRL evaluation awarded partial credit ({matched}/{total} solution steps matched)."
        if answer_eval.decision == "false":
            return "SecRL evaluation: submitted answer did not match the golden answer."
        if answer_eval.error:
            return f"SecRL evaluation failed: {answer_eval.error}"
        return "SecRL evaluation could not confirm correctness."

    def _build_student_learning(
        self,
        score: float,
        answer_eval: _AnswerEvaluation,
        solution_eval: _SolutionEvaluation | None,
    ) -> str:
        if score >= 1.0:
            return "SecRL reward: full credit (answer correct)."
        if solution_eval and solution_eval.reward:
            matched = sum(1 for step in solution_eval.steps or [] if self._interpret_bool(step.get("is_step_correct")))
            total = len(solution_eval.steps or [])
            return f"SecRL reward: partial credit ({matched}/{total} steps correct)."
        if answer_eval.decision == "false":
            return "SecRL reward: answer judged incorrect."
        if answer_eval.error:
            return f"SecRL reward unavailable ({answer_eval.error})."
        return "SecRL reward: no credit awarded."


def build_reward(config: AtlasConfig, reward_config: RewardObjectiveConfig) -> SecRLArcOpsReward:
    params = reward_config.parameters or {}
    llm_cfg = params.get("llm")
    if llm_cfg is None:
        llm_cfg = config.teacher.llm.model_dump()
    answer_llm = LLMParameters.model_validate(llm_cfg)

    step_llm_cfg = params.get("step_llm") or llm_cfg
    step_llm = LLMParameters.model_validate(step_llm_cfg)

    strict = bool(params.get("strict", False))
    step_check = bool(params.get("step_check", True))
    discount = float(params.get("discount_factor", 0.4))

    return SecRLArcOpsReward(
        answer_llm=answer_llm,
        step_llm=step_llm,
        strict=strict,
        step_check=step_check,
        discount_factor=discount,
    )


__all__ = ["build_reward", "SecRLArcOpsReward"]

