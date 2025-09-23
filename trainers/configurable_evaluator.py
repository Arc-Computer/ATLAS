import json
from typing import Any, Dict, List, Optional


class ConfigurableEvaluator:
    """LLM-based evaluation system for ATLAS optimization."""

    def __init__(self, evaluation_config: Optional[Dict[str, Any]] = None):
        self.config = evaluation_config or {}
        self.metrics_config = self.config.get('metrics', [])
        self.evaluation_model = self.config.get('evaluation_model', 'gpt-4.1')
        self.last_metrics = {}
        self.validate_config()

    def validate_config(self):
        """Validate evaluation configuration."""
        if not self.metrics_config:
            self.metrics_config = [{
                'name': 'overall_quality',
                'description': 'Overall quality of the response',
                'weight': 1.0,
                'criteria': 'Evaluate the overall quality and correctness'
            }]

        total_weight = sum(m.get('weight', 0) for m in self.metrics_config)
        if abs(total_weight - 1.0) > 0.01:
            if total_weight > 0:
                for metric in self.metrics_config:
                    metric['weight'] = metric.get('weight', 0) / total_weight

    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """Extract JSON from response text."""
        if not response:
            return None

        try:
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            import re
            matches = re.findall(json_pattern, response, re.DOTALL)

            for match in reversed(matches):
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue

            parsed = json.loads(response)
            if isinstance(parsed, dict):
                return parsed
        except:
            pass

        return None

    def _create_evaluation_prompt(
        self,
        response: str,
        ground_truth: str,
        question: str,
        metric_configs: List[Dict[str, Any]]
    ) -> str:
        """Create detailed evaluation prompt for LLM judge."""

        prompt = """You are an expert evaluator for AI system outputs. Your task is to evaluate the agent's response against the ground truth based on specific criteria.

EVALUATION CONTEXT:
==================
Task/Question:
{question}

Agent's Response:
{response}

Ground Truth (Expected Output):
{ground_truth}

EVALUATION CRITERIA:
===================
You must evaluate based on the following metrics and their weights:

{metrics_description}

SCORING INSTRUCTIONS:
====================
1. For each metric, provide a score from 0.0 to 1.0
2. Consider partial credit where appropriate
3. Be objective and consistent
4. Focus on substance over style
5. Account for equivalent but differently formatted answers

OUTPUT FORMAT:
=============
Return your evaluation as a JSON object with this exact structure:
{{
    "scores": {{
        "metric_name": score_value,
        ...
    }},
    "reasoning": {{
        "metric_name": "Brief explanation for the score",
        ...
    }},
    "final_score": weighted_average_score
}}

IMPORTANT:
- Return ONLY the JSON object, no other text
- All scores must be between 0.0 and 1.0
- The final_score should be the weighted average of all metric scores
"""

        metrics_desc = []
        for metric in metric_configs:
            name = metric.get('name')
            weight = metric.get('weight', 1.0)
            description = metric.get('description', f'Evaluate {name}')
            criteria = metric.get('criteria', '')

            metric_text = f"""
{name} (Weight: {weight:.1%}):
  Description: {description}
  Criteria: {criteria}"""
            metrics_desc.append(metric_text)

        return prompt.format(
            question=question or "No specific question provided",
            response=response,
            ground_truth=ground_truth,
            metrics_description=''.join(metrics_desc)
        )

    def _parse_llm_evaluation(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM evaluation response."""
        try:
            result = self._extract_json_from_response(llm_response)
            if result and 'final_score' in result:
                return result
        except:
            pass

        return {
            'scores': {},
            'reasoning': {},
            'final_score': 0.0
        }

    def evaluate_response(
        self,
        response: str,
        ground_truth: str,
        question: str = ""
    ) -> float:
        """Evaluate a single response using LLM judge."""

        eval_prompt = self._create_evaluation_prompt(
            response=response,
            ground_truth=ground_truth,
            question=question,
            metric_configs=self.metrics_config
        )

        try:
            import litellm

            completion_params = {
                'model': self.evaluation_model,
                'messages': [{"role": "user", "content": eval_prompt}],
            }

            if 'gpt-5' in self.evaluation_model.lower() or 'o1' in self.evaluation_model.lower():
                completion_params['max_completion_tokens'] = 500
                completion_params['reasoning_effort'] = 'low'
            else:
                completion_params['temperature'] = 0.1
                completion_params['max_tokens'] = 500

            llm_response = litellm.completion(**completion_params)

            response_text = llm_response.choices[0].message.content.strip()
            evaluation_result = self._parse_llm_evaluation(response_text)

            self.last_metrics = evaluation_result.get('scores', {})

            final_score = evaluation_result.get('final_score', 0.0)
            final_score = min(max(final_score, 0.0), 1.0)

            return final_score

        except Exception as e:
            print(f"LLM evaluation failed: {e}")
            return 0.0

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        solutions: List[str],
        ground_truths: List[str],
        questions: Optional[List[str]] = None,
    ) -> List[float]:
        """Calculate rewards for a batch of responses."""
        rewards = []

        for i in range(len(solutions)):
            reward = self.evaluate_response(
                response=solutions[i],
                ground_truth=ground_truths[i],
                question=questions[i] if questions else ""
            )
            rewards.append(reward)

        return rewards