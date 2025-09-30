from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class JudgeSpec:
    """Immutable definition for a RIM judge."""

    name: str
    prompt: str


JUDGE_SPECS: Dict[str, JudgeSpec] = {
    "accuracy": JudgeSpec(
        name="accuracy",
        prompt="""Role: Expert solution evaluator. Apply {constitution_principles}.
Task: Evaluate correctness of {student_answer} compared to {ground_truth}.

Context:
- Question/Problem: {question}
- Student's Final Answer: {student_answer}
- Expected Solution: {ground_truth}

For structured outputs (JSON/code), check structural correctness AND semantic accuracy.

Step 1: Generate 2-3 principles for evaluating accuracy.
Consider:
- Correctness of root cause identification
- Accuracy of fault propagation chains
- Completeness of the solution
- Proper format and structure
Assign weight (0.0 to 1.0) to each principle. Weights must sum to 1.0.

Step 2: Evaluate the response against each principle.

Step 3: Provide final score 0.0 to 1.0 based on principle-guided evaluation.
In rationale, state how response performs on each principle and WHY you gave this score.

IMPORTANT: If uncertain, report uncertainty > 0.3.
A larger model will review uncertain cases.

Output JSON: {"principles": [{"name": str, "weight": float, "description": str}], "score": float, "rationale": str, "uncertainty": float}""",
    ),
    "helpfulness": JudgeSpec(
        name="helpfulness",
        prompt="""Role: Teaching effectiveness judge. Apply {constitution_principles}.
Task: Evaluate if the teacher's guidance was helpful based on the student's final performance.

Context:
- Question: {question}
- Student's Initial Plan: {student_plan}
- Teacher's Guidance: {teacher_trace}
- Student's Final Answer: {student_answer}
- Ground Truth: {ground_truth}

IMPORTANT: Compare student's final answer to ground truth. If the answer is incorrect,
consider what the teacher missed or failed to address in their guidance.

Step 1: Generate 2-3 principles for evaluating teaching helpfulness.
Consider:
- Did teacher identify the critical gaps that would lead to correct answer?
- If student's answer is wrong, what did teacher fail to catch or teach?
- Was the guidance specific and actionable enough?
- Did the teaching address the actual problems that led to errors?
Assign weight (0.0 to 1.0) to each principle. Weights must sum to 1.0.

Step 2: Evaluate the teaching against each principle, considering the outcome.
- If student succeeded: Was it because of good teaching?
- If student failed: What did the teacher miss?

Step 3: Provide final score 0.0 to 1.0 based on principle-guided evaluation.
Low scores if teaching missed critical issues that led to wrong answers.
High scores only if teaching addressed the key gaps effectively.

IMPORTANT: If uncertain, report uncertainty > 0.3.
A larger model will review uncertain cases.

Output JSON: {"principles": [{"name": str, "weight": float, "description": str}], "score": float, "rationale": str, "uncertainty": float, "evidence": []}""",
    ),
    "process": JudgeSpec(
        name="process",
        prompt="""Role: Execution trajectory quality judge. Apply {constitution_principles}.
Task: Evaluate how well student executed their plan with teaching guidance.

Context:
- Question: {question}
- Ground Truth: {ground_truth}
- Initial Plan: {student_plan}
- Teaching Guidance: {teacher_trace}
- Execution Trajectory: {student_trace}

The trajectory shows step-by-step what the student actually did during execution.

Step 1: Generate 2-3 principles most relevant for evaluating execution quality.
Consider:
- Did student follow their initial plan?
- Did student apply the teaching guidance effectively?
- Were reasoning steps clear and logical?
- Did student use appropriate tools/methods?
Assign weight (0.0 to 1.0) to each principle. Weights must sum to 1.0.

Step 2: Evaluate the execution trajectory against each principle.

Step 3: Provide final score 0.0 to 1.0 based on principle-guided evaluation.
In rationale, state how execution performs on each principle and WHY you gave this score.

IMPORTANT: If uncertain, report uncertainty > 0.3.
A larger model will review uncertain cases.

Output JSON: {"principles": [{"name": str, "weight": float, "description": str}], "score": float, "rationale": str, "uncertainty": float, "evidence": []}""",
    ),
    "diagnostic": JudgeSpec(
        name="diagnostic",
        prompt="""Role: Student diagnostic quality judge. Apply {constitution_principles}.
Task: Evaluate the quality of the student's initial diagnostic approach and problem-solving plan.

Context:
- Question/Problem: {question}
- Student's Diagnostic Approach: {student_plan}
- Ground Truth (for reference): {ground_truth}

Evaluate how well the student diagnosed the problem and planned their investigation.

Step 1: Generate 2-3 principles for evaluating diagnostic quality.
Consider:
- Did student correctly identify key symptoms and potential root causes?
- Is the diagnostic approach systematic and logical?
- Did student plan to use appropriate tools and methods?
- Are the investigation steps comprehensive and well-reasoned?
Assign weight (0.0 to 1.0) to each principle. Weights must sum to 1.0.

Step 2: Evaluate the student's diagnostic approach against each principle.

Step 3: Provide final score 0.0 to 1.0 based on principle-guided evaluation.
In rationale, state how the diagnostic approach performs on each principle and WHY you gave this score.

IMPORTANT: If uncertain, report uncertainty > 0.3.
A larger model will review uncertain cases.

Output JSON: {"principles": [{"name": str, "weight": float, "description": str}], "score": float, "rationale": str, "uncertainty": float, "evidence": []}""",
    ),
}


def get_judge_prompt(judge_name: str) -> str:
    if judge_name not in JUDGE_SPECS:
        raise ValueError(f"Unknown judge: {judge_name}")
    return JUDGE_SPECS[judge_name].prompt


def parse_judge_payload(payload: Any) -> Optional[Dict[str, Any]]:
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return None
    elif isinstance(payload, dict):
        data = dict(payload)
    else:
        return None

    if data.get('_fallback'):
        data['_fallback'] = True

    score = data.get('score')
    uncertainty = data.get('uncertainty')

    if not isinstance(score, (int, float)) or not isinstance(uncertainty, (int, float)):
        return None

    rationale = data.get('rationale', data.get('explanation', ''))
    principles = data.get('principles', [])
    evidence = data.get('evidence', [])

    if not isinstance(rationale, str):
        rationale = ''
    if not isinstance(principles, list):
        principles = []
    if not isinstance(evidence, list):
        evidence = []

    normalized = {
        'score': float(score),
        'rationale': rationale,
        'uncertainty': float(uncertainty),
        'principles': principles,
        'evidence': evidence,
    }

    for legacy_key in ('score_a', 'score_b', 'explanation'):
        if legacy_key in data:
            normalized[legacy_key] = data[legacy_key]

    if '_fallback' in data:
        normalized['_fallback'] = True

    return normalized
