import json
from pathlib import Path
from string import Template
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union

from gepa.core.adapter import GEPAAdapter, EvaluationBatch
from .extraction_utils import ATLASExtractionUtils


class ATLASDataInst(TypedDict):
    question: str
    ground_truth: str
    additional_context: Optional[Dict[str, str]]


class ATLASTrajectory(TypedDict):
    question: str
    student_approach: str
    teacher_response: str
    student_baseline: str
    student_with_teaching: str
    ground_truth: str
    reward: float


class ATLASRolloutOutput(TypedDict):
    teacher_response: str
    student_with_teaching: str
    student_baseline: str
    reward: float


class ATLASGEPAAdapter(GEPAAdapter[ATLASDataInst, ATLASTrajectory, ATLASRolloutOutput]):

    def _safe_format(self, template_str: str, **kwargs) -> str:
        """
        Safe string formatting that handles content with curly braces.
        Converts {placeholder} syntax to $placeholder for Template class.
        """
        converted_template = template_str
        for key in kwargs:
            placeholder_pattern = '{' + key + '}'
            replacement_pattern = '$' + key
            converted_template = converted_template.replace(placeholder_pattern, replacement_pattern)

        template = Template(converted_template)
        return template.safe_substitute(**kwargs)

    def __init__(
        self,
        teacher_model: Union[str, Callable],
        student_model: Union[str, Callable],
        reward_function: Optional[Callable] = None,
        trace_storage_path: str = "traces/gepa_traces.jsonl",
        all_prompts: Optional[Dict[str, str]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        max_litellm_workers: int = 10,
        use_vllm_client: bool = False,
        vllm_host: Optional[str] = None,
        vllm_port: Optional[int] = None,
    ):
        self.reward_function = reward_function
        self.trace_storage_path = Path(trace_storage_path)
        self.trace_storage_dir = self.trace_storage_path.parent / self.trace_storage_path.stem
        self.trace_storage_dir.mkdir(parents=True, exist_ok=True)
        self.iteration_count = 0
        self.eval_count = 0
        self.max_litellm_workers = max_litellm_workers
        self.use_vllm_client = use_vllm_client
        self.all_prompts = all_prompts or {}
        self.generation_config = generation_config or {}
        
        if use_vllm_client and vllm_host and vllm_port:
            from trainers.vllm_client import VLLMClient
            self.vllm_client = VLLMClient(host=vllm_host, server_port=vllm_port)

            def _create_vllm_generator(client):
                def generate(prompts):
                    is_single = isinstance(prompts, str)
                    if is_single:
                        prompts = [prompts]
                    results = client.generate(
                        prompts=prompts,
                        n=1,
                        temperature=self.generation_config['temperature'],
                        max_tokens=self.generation_config['max_tokens']
                    )
                    return results[0][0] if is_single else [r[0] for r in results]
                return generate

            if isinstance(teacher_model, str):
                self.teacher_model = _create_vllm_generator(self.vllm_client)
            else:
                self.teacher_model = teacher_model

            if isinstance(student_model, str):
                self.student_model = _create_vllm_generator(self.vllm_client)
            else:
                self.student_model = student_model
        else:
            if isinstance(teacher_model, str):
                import litellm
                self.teacher_model = lambda prompts: self._litellm_generate(litellm, teacher_model, prompts)
            else:
                self.teacher_model = teacher_model

            if isinstance(student_model, str):
                import litellm
                self.student_model = lambda prompts: self._litellm_generate(litellm, student_model, prompts)
            else:
                self.student_model = student_model

        self.student_model_str = student_model if isinstance(student_model, str) else None
    
    def _generate_with_student(self, prompts, max_tokens=None):
        if max_tokens is None:
            max_tokens = self.generation_config['max_tokens']
        if self.student_model_str:
            import litellm
            return self._litellm_generate(litellm, self.student_model_str, prompts, max_tokens=max_tokens)
        else:
            return self.student_model(prompts)
    
    def _litellm_generate(
        self,
        litellm,
        model: str,
        prompts: Union[str, List[str]],
        max_tokens: int = None,
        temperature: float = None,
    ) -> Union[str, List[str]]:
        if isinstance(prompts, str):
            prompts = [prompts]
            single = True
        else:
            single = False
        
        if max_tokens is None:
            max_tokens = self.generation_config['max_tokens']
        if temperature is None:
            temperature = self.generation_config['temperature']
        
        print(f"[LiteLLM] Generating for {len(prompts)} prompts with model: {model[:50]}...")
        messages_batch = [[{"role": "user", "content": p}] for p in prompts]
        
        try:
            if model.startswith("https://") and "huggingface.cloud" in model:
                import requests
                import os
                import time
                from concurrent.futures import ThreadPoolExecutor, as_completed

                def make_hf_request(idx, msg):
                    payload = {
                        "inputs": msg[0]["content"],
                        "parameters": {
                            "max_new_tokens": max_tokens,
                            "temperature": temperature,
                            "return_full_text": False
                        }
                    }
                    headers = {"Content-Type": "application/json"}

                    retries = 3
                    for attempt in range(retries):
                        try:
                            response = requests.post(model, json=payload, headers=headers, timeout=120)
                            break
                        except requests.exceptions.Timeout:
                            if attempt < retries - 1:
                                time.sleep(5)
                            else:
                                response = None

                    if response and response.status_code == 200:
                        result = response.json()
                        if isinstance(result, list) and len(result) > 0:
                            return idx, result[0].get("generated_text", "")
                        elif isinstance(result, dict) and "generated_text" in result:
                            return idx, result["generated_text"]
                        else:
                            return idx, str(result)
                    elif response:
                        return idx, f"Error: {response.status_code}"
                    else:
                        return idx, "Error: Timeout"

                print(f"[HF] Processing {len(messages_batch)} requests in parallel with {self.max_litellm_workers} workers...")
                responses = [None] * len(messages_batch)

                with ThreadPoolExecutor(max_workers=self.max_litellm_workers) as executor:
                    futures = {executor.submit(make_hf_request, i, messages): i
                              for i, messages in enumerate(messages_batch)}

                    completed = 0
                    for future in as_completed(futures):
                        idx, result = future.result()
                        responses[idx] = result
                        completed += 1
                        if completed % 50 == 0 or completed == len(messages_batch):
                            print(f"[HF] Completed {completed}/{len(messages_batch)} requests")
            else:
                timeout = self.generation_config.get('timeout', 300)
                request_timeout = self.generation_config.get('request_timeout', 300)

                completion_params = {
                    'model': model,
                    'messages': messages_batch,
                    'max_workers': self.max_litellm_workers,
                    'timeout': timeout,
                    'request_timeout': request_timeout,
                }

                if 'gpt-5' in model.lower() or 'o1' in model.lower():
                    completion_params['max_completion_tokens'] = max_tokens
                    completion_params['reasoning_effort'] = self.generation_config.get('reasoning_effort', 'medium')
                else:
                    completion_params['max_tokens'] = max_tokens
                    completion_params['temperature'] = temperature

                responses = [
                    resp.choices[0].message.content.strip()
                    for resp in litellm.batch_completion(**completion_params)
                ]
            return responses[0] if single else responses
        except Exception as e:
            print(f"Generation error: {str(e)}")
            error_responses = [f"Error: {str(e)}"] * len(prompts)
            return error_responses[0] if single else error_responses
    
    def evaluate(
        self,
        batch: List[ATLASDataInst],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[ATLASTrajectory, ATLASRolloutOutput]:
        
        print(f"\n[Adapter] Evaluating batch of {len(batch)} examples...")
        self.eval_count += 1
        
        self.store_prompt_evolution(candidate)
        
        outputs: List[ATLASRolloutOutput] = []
        scores: List[float] = []
        trajectories: Optional[List[ATLASTrajectory]] = [] if capture_traces else None
        
        questions = [data_inst["question"] for data_inst in batch]
        ground_truths = [data_inst["ground_truth"] for data_inst in batch]
        
        student_diagnostic_template = candidate.get("student_diagnostic_template") or self.all_prompts.get("student_diagnostic_template")
        teacher_adaptive_template = candidate.get("teacher_adaptive_template") or self.all_prompts.get("teacher_adaptive_template")
        student_baseline_template = self.all_prompts.get("student_baseline_template", candidate.get("student_baseline_template", ""))
        student_with_teaching_template = candidate.get("student_with_teaching_template") or self.all_prompts.get("student_with_teaching_template")
        
        baseline_prompts = [
            self._safe_format(student_baseline_template, question=q) for q in questions
        ]
        print(f"[Adapter] Getting baseline solutions from student...")
        baseline_completions = self.student_model(baseline_prompts)
        if not isinstance(baseline_completions, list):
            baseline_completions = [baseline_completions]
        print(f"[Adapter] Got {len(baseline_completions)} baseline solutions")
        
        approach_prompts = [
            self._safe_format(student_diagnostic_template, question=q) for q in questions
        ]
        diagnostic_max_tokens = self.generation_config['diagnostic_max_tokens']
        print(f"[Adapter] Getting student diagnostic approaches (max {diagnostic_max_tokens} tokens)...")
        student_approaches = self._generate_with_student(approach_prompts, max_tokens=diagnostic_max_tokens)
        if not isinstance(student_approaches, list):
            student_approaches = [student_approaches]
        print(f"[Adapter] Got {len(student_approaches)} approaches")
        
        teacher_prompts = [
            self._safe_format(teacher_adaptive_template, question=q, approach=a)
            for q, a in zip(questions, student_approaches)
        ]
        print(f"[Adapter] Getting teacher responses...")
        teacher_completions = self.teacher_model(teacher_prompts)
        if not isinstance(teacher_completions, list):
            teacher_completions = [teacher_completions]
        print(f"[Adapter] Got {len(teacher_completions)} teacher responses")
        
        teaching_contents = [
            ATLASExtractionUtils.extract_teaching_content(tc)
            for tc in teacher_completions
        ]

        student_with_teaching_prompts = [
            self._safe_format(student_with_teaching_template, question=q, teaching=t)
            for q, t in zip(questions, teaching_contents)
        ]
        print(f"[Adapter] Getting student responses with teaching...")
        student_with_teaching_completions = self.student_model(student_with_teaching_prompts)
        if not isinstance(student_with_teaching_completions, list):
            student_with_teaching_completions = [student_with_teaching_completions]
        print(f"[Adapter] Got {len(student_with_teaching_completions)} student responses with teaching")
        
        baseline_solutions = ATLASExtractionUtils.extract_solutions(baseline_completions)
        student_solutions = ATLASExtractionUtils.extract_solutions(student_with_teaching_completions)
        
        if self.reward_function:
            rewards = self.reward_function(
                prompts=questions,
                completions=teacher_completions,
                baseline_solutions=baseline_solutions,
                solutions=student_solutions,
                ground_truths=ground_truths,
            )
        else:
            from .online_teaching_reward import OnlineTeachingReward
            
            class SimpleTokenizer:
                def encode(self, text):
                    return text.split()
            
            default_reward = OnlineTeachingReward(tokenizer=SimpleTokenizer())
            rewards = default_reward(
                prompts=questions,
                completions=teacher_completions,
                baseline_solutions=baseline_solutions,
                solutions=student_solutions,
                ground_truths=ground_truths,
            )
        
        for i in range(len(batch)):
            try:
                output = {
                    "teacher_response": teacher_completions[i] if i < len(teacher_completions) else "",
                    "student_with_teaching": student_with_teaching_completions[i] if i < len(student_with_teaching_completions) else "",
                    "student_baseline": baseline_completions[i] if i < len(baseline_completions) else "",
                    "reward": rewards[i] if i < len(rewards) else 0.0,
                }
                outputs.append(output)
                scores.append(rewards[i] if i < len(rewards) else 0.0)
                
                if capture_traces:
                    trajectory = {
                        "question": questions[i],
                        "student_approach": student_approaches[i] if i < len(student_approaches) else "",
                        "teacher_response": teacher_completions[i] if i < len(teacher_completions) else "",
                        "student_baseline": baseline_completions[i] if i < len(baseline_completions) else "",
                        "student_with_teaching": student_with_teaching_completions[i] if i < len(student_with_teaching_completions) else "",
                        "ground_truth": ground_truths[i],
                        "reward": rewards[i] if i < len(rewards) else 0.0,
                    }
                    trajectories.append(trajectory)
                    self.store_trace(trajectory)
            except Exception as e:
                output = {
                    "teacher_response": f"Error: {str(e)}",
                    "student_with_teaching": "",
                    "student_baseline": "",
                    "reward": 0.0,
                }
                outputs.append(output)
                scores.append(0.0)
                
                if capture_traces:
                    trajectory = {
                        "question": questions[i],
                        "student_approach": f"Error: {str(e)}",
                        "teacher_response": f"Error: {str(e)}",
                        "student_baseline": "",
                        "student_with_teaching": "",
                        "ground_truth": ground_truths[i],
                        "reward": 0.0,
                    }
                    trajectories.append(trajectory)
        
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

        template_roles = {
            "teacher_adaptive_template": {
                "role": "TEACHER/TUTOR",
                "purpose": "Analyzes student approach and provides adaptive teaching guidance that always leads to performance improvement",
                "input": "Question and Student's approach",
                "output": "Teaching guidance wrapped in <teaching> tags"
            },
            "student_diagnostic_template": {
                "role": "STUDENT",
                "purpose": "Shows initial problem-solving approach which tells the teacher how you are going to solve this problem listing all steps. Complete approach under 500 tokens",
                "input": "Question only",
                "output": "Student's step-by-step approach to solving the problem"
            },
            "student_with_teaching_template": {
                "role": "STUDENT",
                "purpose": "Student applies received teaching to complete the problem and provide the final answer within <solution> tags",
                "input": "Question and Teaching received",
                "output": "Student's solution using the teaching, with final answer"
            }
        }

        for component in components_to_update:
            items = []

            template_info = template_roles.get(component, {})
            if template_info:
                metrics_by_template = {
                    "teacher_adaptive_template": {
                        "primary": "Maximize correct answers after teaching (teaching_correct rate)",
                        "secondary": "Ensure teaching is clear, actionable, and directly addresses student's mistakes",
                        "avoid": "Generic advice, overly complex explanations, or teaching that doesn't connect to student's approach"
                    },
                    "student_diagnostic_template": {
                        "primary": "Generate clear diagnostic approaches that reveal student's thinking process",
                        "secondary": "Ensure approach is detailed enough for teacher to identify gaps but concise (under 500 tokens)",
                        "avoid": "Jumping to solution directly, being too vague, or providing complete solutions instead of approach"
                    },
                    "student_with_teaching_template": {
                        "primary": "Maximize correct final answers when applying teaching",
                        "secondary": "Show clear application of teaching concepts, provide answer in proper format (within tags)",
                        "avoid": "Ignoring the teaching, making same mistakes, or failing to provide clear final answer"
                    }
                }

                strategy_by_template = {
                    "teacher_adaptive_template": "Analyze patterns where teaching succeeded vs failed. Identify what made successful teaching effective. Modify prompt to encourage those successful patterns.",
                    "student_diagnostic_template": "Analyze which diagnostic approaches led to most effective teaching. Identify patterns in good vs poor diagnostics. Modify prompt to elicit clearer problem-solving approaches.",
                    "student_with_teaching_template": "Analyze when students successfully applied teaching vs when they didn't. Identify what prompt elements help students integrate teaching. Modify to improve teaching application."
                }

                context_header = {
                    "TEMPLATE_BEING_OPTIMIZED": component,
                    "ROLE": template_info["role"],
                    "PURPOSE": template_info["purpose"],
                    "EXPECTED_INPUT": template_info["input"],
                    "EXPECTED_OUTPUT": template_info["output"],
                    "OPTIMIZATION_GOAL": f"Improve this {template_info['role']} prompt to better achieve: {template_info['purpose']}",
                    "KEY_METRICS_TO_OPTIMIZE": metrics_by_template.get(component, {}),
                    "OPTIMIZATION_STRATEGY": strategy_by_template.get(component, "")
                }
                items.append({"Template Context": context_header})

            for trajectory, score in zip(eval_batch.trajectories, eval_batch.scores):
                baseline_solution = ATLASExtractionUtils.extract_solution(trajectory["student_baseline"])
                teaching_solution = ATLASExtractionUtils.extract_solution(trajectory["student_with_teaching"])

                teaching_content = ATLASExtractionUtils.extract_teaching_content(trajectory["teacher_response"])

                item = {
                    "Inputs": {
                        "question": trajectory["question"],
                        "student_approach": trajectory["student_approach"],
                    },
                    "Generated Outputs": {
                        "teaching_content": teaching_content,
                        "student_with_teaching_solution": teaching_solution,
                    },
                    "Result": {
                        "teaching_correct": ATLASExtractionUtils.check_correctness(teaching_solution, trajectory["ground_truth"]),
                        "expected_answer": trajectory["ground_truth"],
                        "score": score,
                    },
                }

                items.append(item)

            reflective_data[component] = items
        
        return reflective_data
    
    def store_prompt_evolution(self, candidate: Dict[str, str]):
        import time
        prompt_dir = self.trace_storage_dir / "prompt_evolution"
        prompt_dir.mkdir(exist_ok=True)
        
        prompt_data = {
            "eval_count": self.eval_count,
            "timestamp": time.time(),
            "prompts": candidate
        }
        
        filename = f"prompts_eval_{self.eval_count:04d}.json"
        filepath = prompt_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(prompt_data, f, indent=2)
    
    def store_trace(self, trajectory: ATLASTrajectory):
        import time
        trace_data = {
            "eval_count": self.eval_count,
            "timestamp": time.time(),
            "trajectory": trajectory
        }

        filename = f"eval_{self.eval_count:04d}.jsonl"
        filepath = self.trace_storage_dir / filename

        with open(filepath, "a") as f:
            f.write(json.dumps(trace_data) + "\n")