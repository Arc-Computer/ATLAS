from typing import Dict, Any, List
import json


class AccuracyJudge:
    def __init__(self):
        self.name = 'accuracy'

    def evaluate(self, inputs: Dict[str, Any], model_fn, temperature: float) -> Dict[str, Any]:
        prompt = self._build_prompt(inputs)
        response = model_fn(prompt, temperature)
        return self._parse_response(response)

    def _build_prompt(self, inputs: Dict[str, Any]) -> str:
        return f"""Evaluate these two responses to the given prompt.

Prompt: {inputs.get('prompt', '')}

Response A: {inputs.get('response_a', '')}

Response B: {inputs.get('response_b', '')}

Step 1: Generate 2-3 principles that are most relevant for evaluating these responses.
For each principle, assign a weight (percentage from 0.0 to 1.0) based on its importance for this specific query.
The weights should sum to 1.0.

Step 2: Evaluate both responses against each principle you generated.

Step 3: Provide final scores from 0.0 to 1.0 for each response based on your principle-guided evaluation.
In your explanation, clearly state how each response performs on each principle and WHY you gave these final scores.

IMPORTANT: If you have any uncertainty in your evaluation, report a high uncertainty value (>0.5).
A larger, more capable model will review cases where you are uncertain. It is better to escalate
uncertain cases than to provide overconfident scores.

Output JSON only: {{"principles": [{{"name": str, "weight": float, "description": str}}], "score_a": float, "score_b": float, "explanation": str, "uncertainty": float}}"""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        try:
            data = json.loads(response)
            return {
                'principles': data.get('principles', []),
                'score_a': data.get('score_a', 0.0),
                'score_b': data.get('score_b', 0.0),
                'explanation': data.get('explanation', ''),
                'uncertainty': data.get('uncertainty', 0.5)
            }
        except:
            return {'principles': [], 'score_a': 0.0, 'score_b': 0.0, 'explanation': 'Parse error', 'uncertainty': 1.0}


class HelpfulnessJudge:
    def __init__(self):
        self.name = 'helpfulness'

    def evaluate(self, inputs: Dict[str, Any], model_fn, temperature: float) -> Dict[str, Any]:
        prompt = self._build_prompt(inputs)
        response = model_fn(prompt, temperature)
        return self._parse_response(response)

    def _build_prompt(self, inputs: Dict[str, Any]) -> str:
        return f"""Role: Instruction-following judge. Apply {inputs.get('constitution_principles', '')}.
Task: Assess whether teacher guidance caused improvement.

Teacher Trace: {inputs.get('teacher_trace', '')}
Student Baseline: {inputs.get('student_baseline_answer', '')}
Student Final: {inputs.get('student_answer', '')}
Student Trace: {inputs.get('student_trace', '')}

Steps:
1. Identify critical intervention moments
2. Describe delta from baseline to final (accuracy, completeness, clarity)
3. Penalize guidance that added errors or irrelevant detours
4. Map to score_0_1:
   - 1.0: strong causal lift
   - 0.8: moderate lift
   - 0.5: weak/unclear lift
   - 0.2: no lift
   - 0.0: harmful

Output valid JSON only: {{"score": float, "rationale": str, "uncertainty": float, "evidence": []}}"""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        try:
            data = json.loads(response)
            return {
                'score': data.get('score', 0.0),
                'rationale': data.get('rationale', ''),
                'uncertainty': data.get('uncertainty', 0.5),
                'evidence': data.get('evidence', [])
            }
        except:
            return {'score': 0.0, 'rationale': 'Parse error', 'uncertainty': 1.0, 'evidence': []}


class ProcessJudge:
    def __init__(self):
        self.name = 'process'

    def evaluate(self, inputs: Dict[str, Any], model_fn, temperature: float) -> Dict[str, Any]:
        prompt = self._build_prompt(inputs)
        response = model_fn(prompt, temperature)
        return self._parse_response(response)

    def _build_prompt(self, inputs: Dict[str, Any]) -> str:
        return f"""Role: Planner/rubric judge. Apply {inputs.get('constitution_principles', '')}.
Task: Decompose question + ground_truth into required components. Compare with student_plan.

Question: {inputs.get('question', '')}
Ground Truth: {inputs.get('ground_truth', '')}
Student Plan: {inputs.get('student_plan', '')}

Steps:
1. List all required components/constraints
2. Mark present, partially present, missing in student_plan
3. Map to score_0_1:
   - 1.0: all required present & coherent
   - 0.8: minor omissions
   - 0.6: several partials
   - 0.3: major gaps
   - 0.0: off-spec

Output valid JSON only: {{"score": float, "rationale": str, "uncertainty": float, "evidence": []}}"""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        try:
            data = json.loads(response)
            return {
                'score': data.get('score', 0.0),
                'rationale': data.get('rationale', ''),
                'uncertainty': data.get('uncertainty', 0.5),
                'evidence': data.get('evidence', [])
            }
        except:
            return {'score': 0.0, 'rationale': 'Parse error', 'uncertainty': 1.0, 'evidence': []}


class DiagnosticJudge:
    def __init__(self):
        self.name = 'diagnostic'

    def evaluate(self, inputs: Dict[str, Any], model_fn, temperature: float) -> Dict[str, Any]:
        prompt = self._build_prompt(inputs)
        response = model_fn(prompt, temperature)
        return self._parse_response(response)

    def _build_prompt(self, inputs: Dict[str, Any]) -> str:
        return f"""Role: Reasoning critic. Apply {inputs.get('constitution_principles', '')}.
Task: Judge whether the teacher correctly identified root mistakes and explained them precisely.

Student Plan: {inputs.get('student_plan', '')}
Teacher Trace: {inputs.get('teacher_trace', '')}

Steps:
1. Extract primary flaw(s) in student_plan (logic, assumptions, missing steps)
2. Check teacher_trace for correct diagnosis, specificity, and actionable fix
3. Map to score_0_1:
   - 1.0: precise diagnosis + fix
   - 0.8: mostly correct w/ minor gaps
   - 0.5: generic/partial
   - 0.2: misdiagnosed
   - 0.0: wrong + harmful

Output valid JSON only: {{"score": float, "rationale": str, "uncertainty": float, "evidence": []}}"""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        try:
            data = json.loads(response)
            return {
                'score': data.get('score', 0.0),
                'rationale': data.get('rationale', ''),
                'uncertainty': data.get('uncertainty', 0.5),
                'evidence': data.get('evidence', [])
            }
        except:
            return {'score': 0.0, 'rationale': 'Parse error', 'uncertainty': 1.0, 'evidence': []}