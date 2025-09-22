from pathlib import Path
from string import Template
from typing import Any, Callable, Dict, List, Optional, Union

from gepa.core.adapter import GEPAAdapter, EvaluationBatch
from .extraction_utils import ATLASExtractionUtils
from .online_teaching_reward import OnlineTeachingReward
from .prompt_adapter import ATLASDataInst, ATLASTrajectory, ATLASRolloutOutput


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

        if isinstance(teacher_model, str):
            import litellm
            self.teacher_model = lambda prompts: self._litellm_generate(litellm, teacher_model, prompts)

        if isinstance(student_model, str):
            import litellm
            self.student_model = lambda prompts: self._litellm_generate(litellm, student_model, prompts)

    def _safe_format(self, template_str: str, **kwargs) -> str:
        # Simply use format directly - we control the templates
        try:
            return template_str.format(**kwargs)
        except KeyError as e:
            print(f"ERROR: Missing template variable: {e}")
            print(f"Available variables: {list(kwargs.keys())}")
            # Return template with available substitutions
            result = template_str
            for key, value in kwargs.items():
                result = result.replace('{' + key + '}', str(value))
            return result

    def _litellm_generate(self, litellm, model: str, prompts: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(prompts, str):
            prompts = [prompts]
            single = True
        else:
            single = False


        for i, p in enumerate(prompts):
            if not p or p.strip() == "":
                print(f"WARNING: Prompt {i+1} is empty!")
            else:
                print(f"DEBUG: Prompt {i+1} length: {len(p)} chars")
                print(f"DEBUG: Prompt {i+1} preview: {p[:200]}...")

        messages_batch = [[{"role": "user", "content": p}] for p in prompts]

        completion_params = {
            'model': model,
            'messages': messages_batch,
            'max_workers': self.max_litellm_workers,
            'timeout': self.generation_config.get('timeout', 300),
        }

        if 'gpt-5' in model.lower() or 'o1' in model.lower():
            completion_params['max_completion_tokens'] = self.generation_config.get('max_tokens', 2048)
        else:
            completion_params['max_tokens'] = self.generation_config.get('max_tokens', 2048)
            completion_params['temperature'] = self.generation_config.get('temperature', 0.7)

        batch_responses = litellm.batch_completion(**completion_params)

        responses = []
        for i, resp in enumerate(batch_responses):
            if isinstance(resp, Exception):
                print(f"\n❌ Teacher model failed for prompt {i+1}/{len(batch_responses)}:")
                print(f"  Model: {model}")
                print(f"  Error: {resp}")
                raise resp
            else:
                content = resp.choices[0].message.content.strip()
                responses.append(content)

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
            print("❌ ERROR: Required templates not found in candidate")
            print(f"  student_diagnostic_template: {'✓' if student_diagnostic_template else '✗'}")
            print(f"  teacher_adaptive_template: {'✓' if teacher_adaptive_template else '✗'}")
            print(f"  student_with_teaching_template: {'✓' if student_with_teaching_template else '✗'}")
            raise ValueError("Missing required templates in candidate")

        print(f"\n{'='*60}")
        print(f"🔄 GEPA Evaluation #{self.eval_count}")
        print(f"{'='*60}")

        print(f"\n🎓 Step 1: Student provides diagnostic APPROACH (not execution)...")
        approach_prompts = [
            self._safe_format(student_diagnostic_template, question=q)
            for q in questions
        ]

        if self.student_model:
            if callable(self.student_model):
                student_approaches = self.student_model(approach_prompts)
            else:
                student_approaches = approach_prompts
        else:
            student_approaches = self.teacher_model(approach_prompts)

        if not isinstance(student_approaches, list):
            student_approaches = [student_approaches]

        print(f"✅ Student approaches generated. Got {len(student_approaches)} diagnostic approaches")

        # Print student diagnostic response
        for i, approach in enumerate(student_approaches, 1):
            print(f"\n📝 Student Diagnostic Approach #{i}:")
            print("-" * 60)
            print(approach)
            print("-" * 60)

        print(f"\n👨‍🏫 Step 2: Teacher reviews approach and provides teaching...")
        teacher_prompts = [
            self._safe_format(teacher_adaptive_template,
                question=q,
                approach=a)
            for q, a in zip(questions, student_approaches)
        ]

        teacher_responses = self.teacher_model(teacher_prompts)
        if not isinstance(teacher_responses, list):
            teacher_responses = [teacher_responses]

        teaching_contents = [
            ATLASExtractionUtils.extract_teaching_content(tr)
            for tr in teacher_responses
        ]

        print(f"✅ Teaching generated. {len(teacher_responses)} teachings created")

        # Print teacher teaching response
        for i, (full_response, teaching) in enumerate(zip(teacher_responses, teaching_contents), 1):
            print(f"\n👨‍🏫 Teacher Full Response #{i}:")
            print("-" * 60)
            print(full_response)
            print("-" * 60)
            print(f"\n📚 Extracted Teaching #{i}:")
            print("-" * 60)
            print(teaching if teaching else "[NO TEACHING TAGS FOUND - Using full response]")
            print("-" * 60)

        # If no teaching tags found, use full response
        teaching_contents = [
            teaching if teaching else response
            for teaching, response in zip(teaching_contents, teacher_responses)
        ]

        print(f"\n📚 Step 3: Student EXECUTES with teaching...")
        enhanced_prompts = [
            self._safe_format(student_with_teaching_template,
                teaching=teaching,
                question=question,
                approach=approach)
            for teaching, question, approach in zip(teaching_contents, questions, student_approaches)
        ]
        enhanced_responses = self.user_agent(enhanced_prompts)
        if not isinstance(enhanced_responses, list):
            enhanced_responses = [enhanced_responses]

        print(f"✅ Student execution complete. Got {len(enhanced_responses)} responses")

        print(f"\n🔍 Step 4: Running BASELINE agent for comparison...")
        baseline_responses = self.user_agent(questions)
        if not isinstance(baseline_responses, list):
            baseline_responses = [baseline_responses]

        print(f"✅ Baseline complete. Got {len(baseline_responses)} responses")

        baseline_solutions = ATLASExtractionUtils.extract_solutions(baseline_responses)
        enhanced_solutions = ATLASExtractionUtils.extract_solutions(enhanced_responses)

        print(f"\n📊 Step 5: EVALUATING performance...")

        def count_tokens(text):
            return len(text.split())

        baseline_tokens = sum(count_tokens(resp) for resp in baseline_responses)
        enhanced_tokens = sum(count_tokens(resp) for resp in enhanced_responses)
        token_reduction = ((baseline_tokens - enhanced_tokens) / baseline_tokens * 100) if baseline_tokens > 0 else 0

        print(f"\n Token Usage Comparison:")
        print(f"  Baseline tokens: {baseline_tokens}")
        print(f"  With teaching tokens: {enhanced_tokens}")
        print(f"  Difference: {enhanced_tokens - baseline_tokens} ({'+' if enhanced_tokens > baseline_tokens else ''}{token_reduction:.1f}%)")

        if self.evaluation_config:
            from .configurable_evaluator import ConfigurableEvaluator
            reward_calculator = ConfigurableEvaluator(self.evaluation_config)
        else:
            class SimpleTokenizer:
                def encode(self, text):
                    return text.split()
            reward_calculator = OnlineTeachingReward(tokenizer=SimpleTokenizer())

        rewards = reward_calculator(
            prompts=questions,
            completions=teacher_responses,
            baseline_solutions=baseline_solutions,
            solutions=enhanced_solutions,
            ground_truths=ground_truths,
        )

        for i in range(len(batch)):
            output = {
                "student_approach": student_approaches[i],
                "teacher_response": teacher_responses[i],
                "student_with_teaching": enhanced_responses[i],
                "student_baseline": baseline_responses[i],
                "reward": rewards[i],
            }
            outputs.append(output)
            scores.append(rewards[i])

        avg_score = sum(scores) / len(scores) if scores else 0

        print(f"\n{'='*60}")
        print(f"📈 RESULTS for Evaluation #{self.eval_count}")
        print(f"{'='*60}")
        print(f"Average Score: {avg_score:.3f}")

        if hasattr(reward_calculator, 'last_metrics') and reward_calculator.last_metrics:
            print(f"\nMetrics Breakdown:")
            for metric_name, value in reward_calculator.last_metrics.items():
                bar = '█' * int(value * 20) + '░' * (20 - int(value * 20))
                print(f"  {metric_name:<20}: [{bar}] {value:.3f}")

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
            print(f"💾 Saved {len(trajectories)} traces to {trace_file}")

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
                        }
                    }
                    items.append(item)

            if len(items) > 1:
                reflective_data[component] = items

        return reflective_data