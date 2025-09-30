from __future__ import annotations

from typing import Any, Callable, Dict

from RIM.judge_specs import get_judge_prompt, parse_judge_payload


class _BaseJudge:
    def __init__(self, name: str):
        self.name = name

    def evaluate(self, inputs: Dict[str, Any], model_fn: Callable[[str, float], Any], temperature: float) -> Dict[str, Any]:
        prompt = self._build_prompt(inputs)
        response = model_fn(prompt, temperature)
        parsed = parse_judge_payload(response)
        if parsed is None:
            return {
                'score': 0.0,
                'rationale': 'Parse error',
                'uncertainty': 1.0,
                'principles': [],
                'evidence': [],
            }
        return parsed

    def _build_prompt(self, inputs: Dict[str, Any]) -> str:
        template = get_judge_prompt(self.name)
        context = {
            'question': inputs.get('question', inputs.get('prompt', '')),
            'ground_truth': inputs.get('ground_truth', ''),
            'student_answer': inputs.get('student_answer', inputs.get('response_b', '')),
            'student_baseline_answer': inputs.get('student_baseline_answer', inputs.get('response_a', '')),
            'student_plan': inputs.get('student_plan', ''),
            'teacher_trace': inputs.get('teacher_trace', ''),
            'student_trace': inputs.get('student_trace', ''),
            'constitution_principles': inputs.get('constitution_principles', ''),
        }
        return template.format(**context)


class AccuracyJudge(_BaseJudge):
    def __init__(self):
        super().__init__('accuracy')


class HelpfulnessJudge(_BaseJudge):
    def __init__(self):
        super().__init__('helpfulness')


class ProcessJudge(_BaseJudge):
    def __init__(self):
        super().__init__('process')


class DiagnosticJudge(_BaseJudge):
    def __init__(self):
        super().__init__('diagnostic')
