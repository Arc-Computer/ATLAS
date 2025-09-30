import numpy as np
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from RIM.judge_specs import get_judge_prompt, parse_judge_payload
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class RewardSample:
    principles: List[Dict[str, Any]]
    score: float
    explanation: str
    uncertainty: float
    temperature: float


class RewardInterpretationModel:
    def __init__(self, config: Dict[str, Any]):
        self.temperatures = config['temperatures']
        self.variance_threshold = config['variance_threshold']
        self.active_judges = config['active_judges']
        self.models = config['models']
        self.max_workers = config['parallel_execution']['max_workers']

    def evaluate(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        explanations = {}

        active_judges = [judge for judge, active in self.active_judges.items() if active]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_judge = {
                executor.submit(self._evaluate_single_reward, trajectory, judge): judge
                for judge in active_judges
            }

            for future in as_completed(future_to_judge):
                judge = future_to_judge[future]
                reward_result = future.result()
                results[judge] = reward_result['score']
                explanations[judge] = reward_result['explanation']

        return {
            'rewards': results,
            'explanations': explanations
        }

    def _evaluate_single_reward(self, trajectory: Dict[str, Any], reward_type: str) -> Dict[str, Any]:
        samples = []

        with ThreadPoolExecutor(max_workers=len(self.temperatures)) as executor:
            futures = [
                executor.submit(self._run_small_model, trajectory, reward_type, temp)
                for temp in self.temperatures
            ]
            for future in as_completed(futures):
                sample = future.result()
                if sample is not None:
                    samples.append(sample)

        if not samples:
            return self._escalate_to_large_model(
                trajectory,
                reward_type,
                samples=[],
                failure_reason="tier1_no_valid_samples",
            )

        scores = [s.score for s in samples]
        uncertainties = [s.uncertainty for s in samples]

        variance = np.std(scores)
        max_uncertainty = max(uncertainties)

        should_escalate = (
            variance > self.variance_threshold or
            max_uncertainty > 0.3
        )

        if not should_escalate:
            best_sample = min(samples, key=lambda x: x.uncertainty)
            return {
                'score': best_sample.score,
                'explanation': best_sample.explanation
            }
        else:
            return self._escalate_to_large_model(trajectory, reward_type, samples)

    def _run_small_model(self, trajectory: Dict[str, Any], reward_type: str, temperature: float) -> RewardSample | None:
        small_model = self.models['small_model']

        prompt = self._build_judge_prompt(trajectory, reward_type)
        response = self._normalize_model_output(
            self._call_model(small_model, prompt, temperature)
        )

        if response is None:
            return None

        score = response.get('score')
        uncertainty = response.get('uncertainty')
        explanation = response.get('rationale', response.get('explanation', ''))
        principles = response.get('principles', [])

        if response.get('_fallback'):
            return None

        if not isinstance(score, (int, float)) or not isinstance(uncertainty, (int, float)):
            return None

        if not isinstance(explanation, str):
            explanation = ''
        if not isinstance(principles, list):
            principles = []

        return RewardSample(
            principles=principles,
            score=score,
            explanation=explanation,
            uncertainty=uncertainty,
            temperature=temperature
        )

    def _escalate_to_large_model(
        self,
        trajectory: Dict[str, Any],
        reward_type: str,
        samples: List[RewardSample],
        failure_reason: str | None = None,
    ) -> Dict[str, Any]:
        large_model = self.models['large_model']

        meta_prompt = self._build_meta_prompt(trajectory, reward_type, samples, failure_reason)
        response = self._normalize_model_output(
            self._call_model(large_model, meta_prompt, 0.3)
        )

        if response is None:
            return {
                'score': 0.0,
                'explanation': 'Large-judge escalation failed to produce a valid response.'
                + (f" Reason: {failure_reason}" if failure_reason else ''),
            }

        score = response.get('score')
        if not isinstance(score, (int, float)):
            score = 0.0

        explanation = response.get('rationale', response.get('explanation', ''))
        if not isinstance(explanation, str):
            explanation = ''

        if failure_reason:
            explanation = f"[{failure_reason}] {explanation}".strip()

        return {
            'score': score,
            'explanation': explanation
        }

    def _normalize_model_output(self, output: Any) -> Dict[str, Any] | None:
        return parse_judge_payload(output)

    def _build_judge_prompt(self, trajectory: Dict[str, Any], reward_type: str) -> str:
        base_prompt = self._get_base_prompt(reward_type)
        return base_prompt.format(
            question=trajectory.get('question', ''),
            ground_truth=trajectory.get('ground_truth', ''),
            student_answer=trajectory.get('student_answer', ''),
            student_baseline_answer=trajectory.get('student_baseline_answer', ''),
            student_plan=trajectory.get('student_plan', ''),
            teacher_trace=trajectory.get('teacher_trace', ''),
            student_trace=trajectory.get('student_trace', ''),
            constitution_principles=trajectory.get('constitution_principles', '')
        )

    def _get_base_prompt(self, reward_type: str) -> str:
        return get_judge_prompt(reward_type)

    def _build_meta_prompt(
        self,
        trajectory: Dict[str, Any],
        reward_type: str,
        samples: List[RewardSample],
        failure_reason: str | None = None,
    ) -> str:
        if samples:
            samples_text = "\n\n".join([
                f"Evaluation {i+1}:\nPrinciples: {json.dumps(s.principles)}\nScore: {s.score:.2f}\nUncertainty: {s.uncertainty:.2f}\nRationale: {s.explanation}"
                for i, s in enumerate(samples)
            ])
        else:
            if failure_reason:
                samples_text = (
                    "Tier-1 judges failed to produce parsable outputs."
                    " Escalating directly to the arbiter model."
                )
            else:
                samples_text = "Tier-1 judge outputs unavailable."

        return f"""Previous evaluations show disagreement or uncertainty about this response.

Tier-1 evaluations (each includes principles, score, uncertainty, rationale):

{samples_text}

Original context:
Question: {trajectory.get('question', '')}
Ground Truth: {trajectory.get('ground_truth', '')}
Student Answer: {trajectory.get('student_answer', '')}
Student Baseline Answer: {trajectory.get('student_baseline_answer', '')}
Student Plan: {trajectory.get('student_plan', '')}
Teacher Trace: {trajectory.get('teacher_trace', '')}
Student Trace: {trajectory.get('student_trace', '')}

Steps:
1. Review all principles generated by the three evaluations and identify which are most valid
2. Read all three rationales to understand their reasoning
3. Read the student response and context carefully
4. Determine if previous evaluations are correct, partially correct, or incorrect
5. Generate your own final set of principles if needed, or synthesize the best principles
6. Provide your final score 0.0 to 1.0 based on the best principles

In your rationale, state:
- Which principles from the three evaluations are most valid or if you generated your own
- WHY you chose this final score after considering all principles and evaluations
- How your final judgment aligns or differs from the previous evaluations

Output JSON: {{"principles": [{{"name": str, "weight": float, "description": str}}], "score": float, "rationale": str, "uncertainty": float}}"""

    def _call_model(self, model_name: str, prompt: str, temperature: float) -> Dict[str, Any]:
        from RIM.model_interface import model_interface

        response = model_interface.call_model(
            model_name=model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=32768
        )

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                'score': 0.0,
                'rationale': 'Failed to parse model response',
                'uncertainty': 1.0,
                '_fallback': True,
            }
