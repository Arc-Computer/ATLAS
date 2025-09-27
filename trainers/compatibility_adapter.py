from pathlib import Path
from string import Template
from typing import Any, Callable, Dict, List, Optional, Union

from gepa.core.adapter import GEPAAdapter, EvaluationBatch
from .extraction_utils import ATLASExtractionUtils
from RIM.reward_adapter import RIMReward
from .prompt_adapter import ATLASDataInst, ATLASTrajectory, ATLASRolloutOutput
from .terminal_display import DisplayManager


class CompatibilityAdapter(GEPAAdapter[ATLASDataInst, ATLASTrajectory, ATLASRolloutOutput]):
    """Adapter for testing existing agents with ATLAS teaching framework."""

    def __init__(
        self,
        teacher_model: Union[str, Callable],
        user_agent: Callable,
        trace_storage_path: str = "traces/compatibility_traces.jsonl",
        generation_config: Optional[Dict[str, Any]] = None,
        max_litellm_workers: int = 10,
        reflection_instructions: Optional[Dict[str, str]] = None,
        evaluation_config: Optional[Dict[str, Any]] = None,
        optimization_targets: Optional[Dict[str, Any]] = None,
        student_model: Optional[Union[str, Callable]] = None,
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.user_agent = user_agent
        self.trace_storage_path = Path(trace_storage_path)
        self.trace_storage_dir = self.trace_storage_path.parent / self.trace_storage_path.stem
        self.trace_storage_dir.mkdir(parents=True, exist_ok=True)
        self.eval_count = 0
        self.generation_config = generation_config or {}
        self.max_litellm_workers = max_litellm_workers
        self.reflection_instructions = reflection_instructions or {}
        self.evaluation_config = evaluation_config or {}
        self.optimization_targets = optimization_targets or {}
        self.display_manager = DisplayManager(verbose=True)
        self.total_evaluations = 0

        if isinstance(teacher_model, str):
            import litellm
            self.teacher_model = lambda prompts: self._litellm_generate(litellm, teacher_model, prompts)

        if isinstance(student_model, str):
            import litellm
            self.student_model = lambda prompts: self._litellm_generate(litellm, student_model, prompts)

    def _safe_format(self, template_str: str, **kwargs) -> str:
        try:
            return template_str.format(**kwargs)
        except (KeyError, IndexError) as e:
            if isinstance(e, IndexError):
                print(f"ERROR: Template has numbered placeholders but got named arguments")
            else:
                print(f"ERROR: Missing template variable: {e}")
            print(f"Available variables: {list(kwargs.keys())}")
            result = template_str
            for key, value in kwargs.items():
                result = result.replace('{' + key + '}', str(value))
            return result

    def _litellm_generate(self, litellm, model: str, prompts: Union[str, List[str]]) -> Union[str, List[str]]:
        import logging
        import os

        os.environ["LITELLM_LOG"] = "ERROR"
        litellm.suppress_debug_info = True
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("openai").setLevel(logging.ERROR)

        if isinstance(prompts, str):
            prompts = [prompts]
            single = True
        else:
            single = False

        if model.startswith("https://") and "huggingface" in model:
            return self._call_hf_endpoint(model, prompts, single)

        messages_batch = [[{"role": "user", "content": p}] for p in prompts]

        completion_params = {
            'model': model,
            'messages': messages_batch,
            'max_workers': self.max_litellm_workers,
            'timeout': self.generation_config.get('timeout', 300),
        }

        if 'gpt-5' in model.lower() or 'o1' in model.lower():
            completion_params['max_completion_tokens'] = self.generation_config.get('max_tokens', 2048)
            completion_params['reasoning_effort'] = self.generation_config.get('reasoning_effort', 'medium')
        else:
            completion_params['max_tokens'] = self.generation_config.get('max_tokens', 2048)
            completion_params['temperature'] = self.generation_config.get('temperature', 0.7)

        batch_responses = litellm.batch_completion(**completion_params)

        responses = []
        for i, resp in enumerate(batch_responses):
            if isinstance(resp, Exception):
                raise resp
            else:
                content = resp.choices[0].message.content.strip()
                responses.append(content)

        return responses[0] if single else responses

    def _call_hf_endpoint(self, endpoint_url: str, prompts: List[str], single: bool) -> Union[str, List[str]]:
        import requests
        import time

        responses = []
        for prompt in prompts:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": self.generation_config.get('max_tokens', 200),
                    "temperature": self.generation_config.get('temperature', 0.7),
                    "return_full_text": False
                }
            }

            headers = {"Content-Type": "application/json"}
            max_retries = 3

            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        endpoint_url,
                        headers=headers,
                        json=payload,
                        timeout=self.generation_config.get('timeout', 300)
                    )

                    if response.status_code == 200:
                        result = response.json()
                        if isinstance(result, list):
                            text = result[0].get("generated_text", "")
                        else:
                            text = result.get("generated_text", "")
                        responses.append(text)
                        break
                    else:
                        if attempt < max_retries - 1:
                            time.sleep(2)
                        else:
                            responses.append(f"Error: HF endpoint returned {response.status_code}")
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        responses.append(f"Error calling HF endpoint: {e}")

        return responses[0] if single else responses


    def evaluate(
        self,
        batch: List[ATLASDataInst],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[ATLASTrajectory, ATLASRolloutOutput]:

        self.eval_count += 1
        outputs: List[ATLASRolloutOutput] = []
        scores: List[float] = []
        trajectories: Optional[List[ATLASTrajectory]] = [] if capture_traces else None

        questions = [data_inst["question"] for data_inst in batch]
        ground_truths = [data_inst["ground_truth"] for data_inst in batch]

        student_diagnostic_template = candidate.get("student_diagnostic_template", "")
        teacher_adaptive_template = candidate.get("teacher_adaptive_template", "")
        student_with_teaching_template = candidate.get("student_with_teaching_template", "")

        if not student_diagnostic_template or not teacher_adaptive_template or not student_with_teaching_template:
            raise ValueError("Missing required templates in candidate")

        if not hasattr(self.display_manager, 'display') or not self.display_manager.display:
            total_evals = getattr(self, 'total_evaluations', 50)
            self.display_manager.start(total_evals)

        approach_prompts = [
            self._safe_format(student_diagnostic_template, question=q)
            for q in questions
        ]

        if self.display_manager:
            self.display_manager.update("student", text="")

        if self.student_model:
            if callable(self.student_model):
                student_approaches = self.student_model(approach_prompts)
            else:
                student_approaches = approach_prompts
        else:
            student_approaches = self.teacher_model(approach_prompts)

        if not isinstance(student_approaches, list):
            student_approaches = [student_approaches]

        if student_approaches and self.display_manager:
            self.display_manager.update("student",
                text=student_approaches[0],
                sample=1,
                total=len(student_approaches))

        teacher_prompts = [
            self._safe_format(teacher_adaptive_template,
                question=q,
                approach=a)
            for q, a in zip(questions, student_approaches)
        ]

        if self.display_manager:
            self.display_manager.update("teacher", text="")

        teacher_responses = self.teacher_model(teacher_prompts)
        if not isinstance(teacher_responses, list):
            teacher_responses = [teacher_responses]

        teaching_contents = [
            ATLASExtractionUtils.extract_teaching_content(tr)
            for tr in teacher_responses
        ]

        teaching_contents = [
            teaching if teaching else response
            for teaching, response in zip(teaching_contents, teacher_responses)
        ]

        if teaching_contents and self.display_manager:
            self.display_manager.update("teacher",
                text=teaching_contents[0],
                sample=1,
                total=len(teaching_contents))

        enhanced_prompts = [
            self._safe_format(student_with_teaching_template,
                teaching=teaching,
                question=question,
                approach=approach)
            for teaching, question, approach in zip(teaching_contents, questions, student_approaches)
        ]
        if self.display_manager:
            self.display_manager.update("student_with_teaching", text="")

        enhanced_responses = self.user_agent(enhanced_prompts)
        if not isinstance(enhanced_responses, list):
            enhanced_responses = [enhanced_responses]

        if enhanced_responses and self.display_manager:
            self.display_manager.update("student_with_teaching", text=str(enhanced_responses[0]))

        if self.display_manager:
            self.display_manager.update("baseline", text="")

        baseline_responses = self.user_agent(questions)
        if not isinstance(baseline_responses, list):
            baseline_responses = [baseline_responses]

        if baseline_responses and self.display_manager:
            self.display_manager.update("baseline", text=str(baseline_responses[0]))

        baseline_solutions = ATLASExtractionUtils.extract_solutions(baseline_responses)
        enhanced_solutions = ATLASExtractionUtils.extract_solutions(enhanced_responses)

        def count_tokens(text):
            return len(text.split())

        baseline_tokens = sum(count_tokens(resp) for resp in baseline_responses)
        enhanced_tokens = sum(count_tokens(resp) for resp in enhanced_responses)
        token_reduction = ((baseline_tokens - enhanced_tokens) / baseline_tokens * 100) if baseline_tokens > 0 else 0

        reward_calculator = RIMReward(config_path='configs/rim_config.yaml')

        _, info_dicts = reward_calculator(
            prompts=questions,
            completions=teacher_responses,
            ground_truths=ground_truths,
            student_plans=[approach for approach in student_approaches],
            teacher_traces=teacher_responses,
            student_traces=enhanced_responses,
            return_info_dict=True,
        )

        detailed_metrics = {}
        for judge in ['accuracy', 'helpfulness', 'process', 'diagnostic']:
            judge_rewards = [info['rewards'].get(judge, 0.0) for info in info_dicts]
            if judge_rewards:
                detailed_metrics[judge] = sum(judge_rewards) / len(judge_rewards)

        for i in range(len(batch)):
            rim_rewards = info_dicts[i]['rewards']

            combined_score = (
                rim_rewards.get('accuracy', 0.0) +
                rim_rewards.get('helpfulness', 0.0) +
                rim_rewards.get('process', 0.0) +
                rim_rewards.get('diagnostic', 0.0)
            ) / 4

            output = {
                "student_approach": student_approaches[i],
                "teacher_response": teacher_responses[i],
                "student_with_teaching": enhanced_responses[i],
                "student_baseline": baseline_responses[i],
                "combined_score": combined_score,
                "rim_rewards": rim_rewards,
                "rim_explanations": info_dicts[i]['explanations'],
            }
            outputs.append(output)
            scores.append(combined_score)

        avg_score = sum(scores) / len(scores) if scores else 0

        metrics_dict = {
            "avg_reward": avg_score,
            "token_savings": token_reduction
        }


        if self.display_manager:
            self.display_manager.update("iteration_complete",
                iteration=self.eval_count,
                score=avg_score,
                metrics=metrics_dict)
            self.display_manager.update("progress", current=self.eval_count, total=getattr(self, 'total_evaluations', 50))

        if capture_traces:
            for i in range(len(batch)):
                trajectory = {
                    "question": questions[i],
                    "student_approach": student_approaches[i],
                    "teacher_response": teacher_responses[i],
                    "student_baseline": baseline_responses[i],
                    "student_with_teaching": enhanced_responses[i],
                    "ground_truth": ground_truths[i],
                    "reward": rewards[i],
                    "token_usage": {
                        "baseline_tokens": count_tokens(baseline_responses[i]),
                        "enhanced_tokens": count_tokens(enhanced_responses[i]),
                        "reduction_percent": ((count_tokens(baseline_responses[i]) - count_tokens(enhanced_responses[i])) / count_tokens(baseline_responses[i]) * 100) if count_tokens(baseline_responses[i]) > 0 else 0
                    }
                }
                trajectories.append(trajectory)

            import json
            import datetime
            trace_file = self.trace_storage_path
            with open(trace_file, 'a') as f:
                for traj in trajectories:
                    trace_entry = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "eval_count": self.eval_count,
                        "trajectory": traj
                    }
                    f.write(json.dumps(trace_entry) + '\n')

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_batch: EvaluationBatch[ATLASTrajectory, ATLASRolloutOutput],
        components_to_update: List[str],
    ) -> Dict[str, List[Dict[str, Any]]]:

        reflective_data = {}

        for component in components_to_update:
            target_config = self.optimization_targets.get(component, {})

            if not target_config.get('optimize', True):
                continue

            items = []

            default_goals = {
                'teacher_adaptive_template': "Improve teaching clarity and relevance based on student's errors.",
                'student_diagnostic_template': "Better identify student misconceptions and knowledge gaps.",
                'student_with_teaching_template': "Optimize how teaching guidance is integrated into responses."
            }

            goal = target_config.get('reflection_goal') or \
                   self.reflection_instructions.get(component) or \
                   default_goals.get(component, f"Optimize {component} for better performance")

            items.append({
                "OPTIMIZATION_TARGET": component,
                "GOAL": goal
            })

            if eval_batch.trajectories:
                for trajectory, score in zip(eval_batch.trajectories, eval_batch.scores):
                    teacher_response = trajectory.get("teacher_response", "")
                    teaching_content = ATLASExtractionUtils.extract_teaching_content(teacher_response)

                    rim_rewards = trajectory.get("rim_rewards", {})
                    rim_explanations = trajectory.get("rim_explanations", {})

                    item = {
                        "Inputs": {
                            "question": trajectory["question"],
                            "student_diagnostic_approach": trajectory["student_approach"],
                        },
                        "Teaching": {
                            "teacher_guidance": teaching_content,
                        },
                        "Output": {
                            "student_response_with_teaching": trajectory["student_with_teaching"],
                        },
                        "Expected": {
                            "ground_truth": trajectory["ground_truth"],
                        },
                        "Performance": {
                            "score": score,
                            "rim_rewards": rim_rewards,
                            "rim_explanations": rim_explanations,
                        }
                    }
                    items.append(item)

            if len(items) > 1:
                reflective_data[component] = items

        return reflective_data