import os
import abc
import gc
import copy
import json
import torch
import accelerate
from torch import nn
import torch.nn.functional as F
from typing import Any, Callable, Optional, Union
from transformers import PreTrainedTokenizer
from .grpo import GRPOTrainer
from .grpo_config import GRPOConfig
from .teacher_base import TeacherReward, TeacherTrainer
from .utils_trl_15 import prepare_deepspeed, unwrap_model_for_generation
from transformers import AutoModelForCausalLM
from trl.data_utils import maybe_apply_chat_template, is_conversational
from accelerate.utils import gather, gather_object, broadcast_object_list


class TeacherGRPOTrainer(GRPOTrainer, TeacherTrainer):
    def __init__(
            self,
            *args,
            student_model=None,


            use_reference_teacher_model=False,
            student_model_init_kwargs=None,
            logging_prob=0.0,


            disable_student_offloading=False,
            **kwargs):

        GRPOTrainer.__init__(self, *args, **kwargs)
        if student_model_init_kwargs is None:
            student_model_init_kwargs = self._stored_model_init_kwargs

        offload_student_model = self.offload_untrained_models and (
            not disable_student_offloading)
        if student_model is None:

            self.student_model = self.ref_model
        elif isinstance(student_model, str):
            self.student_model = AutoModelForCausalLM.from_pretrained(
                student_model, **student_model_init_kwargs)
            if self.is_deepspeed_enabled:
                self.student_model = prepare_deepspeed(
                    self.student_model,
                    self.accelerator,
                    offload_to_cpu=offload_student_model)
            else:
                self.student_model = self.accelerator.prepare_model(
                    self.student_model, evaluation_mode=True)

                if offload_student_model:
                    self.student_model = accelerate.cpu_offload(
                        model=self.student_model)
        else:

            raise NotImplementedError
            self.student_model = student_model

        if use_reference_teacher_model:
            teacher_model = self.ref_model
        else:
            teacher_model = self.model

        TeacherTrainer.__init__(
            self,
            student_model=self.student_model,
            teacher_model=teacher_model,
            tokenizer=self.processing_class,
            reward_functions=self.reward_funcs,
            output_dir=self.args.output_dir,
            logging_prob=logging_prob,
        )
        
        self.max_probe_tokens = getattr(self.args, 'max_probe_tokens', 500)
        self.student_diagnostic_template = getattr(self.args, 'student_diagnostic_template',
                                                   "Question: {question}\n\nBriefly outline your approach:")
        self.teacher_adaptive_template = getattr(self.args, 'teacher_adaptive_template',
                                                "Question: {question}\n\nStudent's approach: {approach}\n\nProvide adaptive teaching:")
        self.student_with_teaching_template = getattr(self.args, 'student_with_teaching_template',
                                                     "Question: {question}\n\nTeaching: {teaching}\n\nSolve the problem:")
        self.student_baseline_template = getattr(self.args, 'student_baseline_template',
                                                "Question: {question}\n\nSolve this problem:")
        
        if self.use_vllm_server and self.accelerator.is_main_process:
            if hasattr(self, 'vllm_clients'):
                teacher_client_indices = [i for i in range(len(self.vllm_clients)) if i % 2 == 0]
                student_client_indices = [i for i in range(len(self.vllm_clients)) if i % 2 == 1]
                
                self.teacher_vllm_clients = [self.vllm_clients[i] for i in teacher_client_indices]
                self.student_vllm_clients = [self.vllm_clients[i] for i in student_client_indices]
            else:
                raise RuntimeError("Parent GRPOTrainer did not initialize vllm_clients")
    
    def _generate_and_score(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)[
            "prompt"] for example in inputs]

        step_contexts = [example.get("step_context") or {} for example in inputs]
        guidance_histories = [example.get("guidance_history") or [] for example in inputs]
        validations = [example.get("validation") or {} for example in inputs]
        tool_names = [example.get("tool") for example in inputs]
        tool_parameters = [example.get("tool_params") or {} for example in inputs]
        reward_breakdowns = [example.get("reward_breakdown") for example in inputs]
        reward_scores = [example.get("reward_score") for example in inputs]

        def _pretty(value):
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            try:
                return json.dumps(value, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                return str(value)
        
        self._print_debugging_logs('Generating baseline solutions...')
        baseline_completions_text = self._generate_baseline_solutions(prompts_text)
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()
        
        self._print_debugging_logs('Generating student approaches...')
        student_approaches = self._generate_student_approaches(prompts_text)
        gc.collect()
        torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()
        
        def _safe_format(template: str, **values: str) -> str:
            try:
                return template.format(**values)
            except KeyError as exc:
                return template + "\n" + "\n".join(f"{key}: {values[key]}" for key in sorted(values))

        self._print_debugging_logs('Generating teacher adaptive teaching...')
        teacher_prompts = []
        for idx, (prompt, approach) in enumerate(zip(prompts_text, student_approaches)):
            teacher_prompts.append(
                _safe_format(
                    self.teacher_adaptive_template,
                    question=prompt,
                    approach=approach,
                    step_context=_pretty(step_contexts[idx]),
                    guidance_history=_pretty(guidance_histories[idx]),
                    validation=_pretty(validations[idx]),
                    tool=tool_names[idx] or "none",
                    tool_params=_pretty(tool_parameters[idx]),
                )
            )
        
        if self.args.use_vllm or self.args.use_ray or self.args.use_vllm_server:
            teacher_completion_ids = self._generate_with_vllm(teacher_prompts, role="teacher")
        else:
            teacher_completion_ids = self._generate_with_model(self.model, teacher_prompts)
        
        teacher_completions_text = self.processing_class.batch_decode(
            teacher_completion_ids, skip_special_tokens=True)
        gc.collect()
        torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()
        
        self._print_debugging_logs('Generating student solutions with teaching...')
        student_prompts_with_teaching = []
        teaching_only_text = []
        for idx, (prompt, full_teaching) in enumerate(zip(prompts_text, teacher_completions_text)):
            import re
            teaching_match = re.search(r'<teaching>(.*?)</teaching>', full_teaching, re.DOTALL)
            if teaching_match:
                teaching_content = teaching_match.group(1).strip()
            else:
                teaching_content = full_teaching.strip()

            teaching_only_text.append(teaching_content)
            student_prompts_with_teaching.append(
                _safe_format(
                    self.student_with_teaching_template,
                    question=prompt,
                    teaching=teaching_content,
                    step_context=_pretty(step_contexts[idx]),
                    guidance_history=_pretty(guidance_histories[idx]),
                    validation=_pretty(validations[idx]),
                    tool=tool_names[idx] or "none",
                    tool_params=_pretty(tool_parameters[idx]),
                )
            )
        
        if self.args.use_vllm or self.args.use_ray or self.args.use_vllm_server:
            student_completion_ids = self._generate_with_vllm(student_prompts_with_teaching)
        else:
            student_completion_ids = self._generate_with_model(self.student_model, student_prompts_with_teaching)
        
        student_completions_text = self.processing_class.batch_decode(
            student_completion_ids, skip_special_tokens=True)
        gc.collect()
        torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()
        
        student_solutions = self._extract_solutions_from_responses(student_completions_text)
        baseline_solutions = self._extract_solutions_from_responses(baseline_completions_text)
        
        completions = teacher_completions_text
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, teacher_completions_text):
                bootstrap = prompt.pop()[
                    "content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}])
        
        self._print_debugging_logs('computing rewards...')
        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device)
        
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c}
                                for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)[
                        "text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(
                        **reward_inputs).logits[:, 0]
            else:
                keys = [key for key in inputs[0]
                        if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key]
                                     for example in inputs] for key in keys}

                reward_kwargs['baseline_solutions'] = baseline_solutions
                reward_kwargs['solutions'] = student_solutions
                reward_kwargs['ground_truths'] = [example.get('ground_truth', '') for example in inputs]
                reward_kwargs.setdefault('guidance_history', guidance_histories)
                reward_kwargs.setdefault('step_context', step_contexts)
                reward_kwargs.setdefault('validation', validations)

                output_reward_func = reward_func(
                    prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(
                    output_reward_func, dtype=torch.float32, device=device)
                
                if self.accelerator.is_main_process:
                    for j, (reward_val, prompt, full_teaching, student_sol, baseline_sol, gt, teaching_only) in enumerate(
                        zip(output_reward_func, prompts_text, completions, 
                            student_solutions, baseline_solutions, 
                            reward_kwargs['ground_truths'], teaching_only_text)
                    ):
                        if j < 3:
                            print(f"\n[REWARD CHECK] Sample {j}:")
                            print(f"  Ground Truth: {gt}")
                            print(f"  Baseline Solution: {baseline_sol}")
                            print(f"  Student Solution: {student_sol}")
                            print(f"  Reward: {reward_val}")
                        
                        if reward_val >= 1.0:
                            import os
                            import json
                            import time
                            success_dir = os.path.join(self.args.output_dir, "successful_teaching")
                            os.makedirs(success_dir, exist_ok=True)
                            timestamp = int(time.time() * 1000000)
                            
                            success_example = {
                                "step": self.state.global_step,
                                "timestamp": timestamp,
                                "sample_index": j,
                                "prompt": prompt,
                                "student_approach": student_approaches[j] if j < len(student_approaches) else "",
                                "teaching_content": teaching_only,
                                "full_teacher_response": full_teaching,
                                "student_solution": student_sol,
                                "baseline_solution": baseline_sol,
                                "ground_truth": gt,
                                "reward": float(reward_val),
                                "prior_reward": reward_scores[j] if j < len(reward_scores) else None,
                                "reward_breakdown": reward_breakdowns[j] if j < len(reward_breakdowns) else None,
                                "guidance_history": guidance_histories[j] if j < len(guidance_histories) else [],
                                "step_context": step_contexts[j] if j < len(step_contexts) else {},
                                "student_full_response": student_completions_text[j] if j < len(student_completions_text) else "",
                                "baseline_full_response": baseline_completions_text[j] if j < len(baseline_completions_text) else ""
                            }
                            
                            filename = f"{success_dir}/step_{self.state.global_step:06d}_sample_{j}_{timestamp}.json"
                            with open(filename, 'w') as f:
                                json.dump(success_example, f, indent=2, ensure_ascii=False)
                            print(f"  [SAVED] High-reward teaching saved to {filename}")
        
        self._print_debugging_logs('gathering rewards...')
        
        rewards_per_func = gather(rewards_per_func)
        
        rewards = (rewards_per_func *
                   self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        
        self._print_debugging_logs('normalizing rewards...')
        
        mean_grouped_rewards = rewards.view(-1,
                                           self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(
                reward_per_func[i].item())
        
        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())
        
        ground_truths = [example.get("ground_truth", "") for example in inputs]
        
        student_correct = 0
        baseline_correct = 0
        teacher_lengths = []
        improvement_count = 0
        degradation_count = 0
        
        for i, (gt, student_sol, baseline_sol, teacher_text) in enumerate(
            zip(ground_truths, student_solutions, baseline_solutions, teacher_completions_text)
        ):
            if gt:
                student_is_correct = self._check_solution_correctness(student_sol, gt)
                baseline_is_correct = self._check_solution_correctness(baseline_sol, gt)
                
                if student_is_correct:
                    student_correct += 1
                if baseline_is_correct:
                    baseline_correct += 1
                    
                if student_is_correct and not baseline_is_correct:
                    improvement_count += 1
                elif not student_is_correct and baseline_is_correct:
                    degradation_count += 1
                    
            teaching_tokens = len(self.processing_class.encode(teacher_text))
            teacher_lengths.append(teaching_tokens)
        
        total_samples = len(ground_truths)
        samples_with_gt = sum(1 for gt in ground_truths if gt)
        
        if samples_with_gt > 0:
            self._metrics["student_accuracy"].append(student_correct / samples_with_gt)
            self._metrics["baseline_accuracy"].append(baseline_correct / samples_with_gt)
            self._metrics["accuracy_delta"].append((student_correct - baseline_correct) / samples_with_gt)
            self._metrics["improvement_rate"].append(improvement_count / samples_with_gt)
            self._metrics["degradation_rate"].append(degradation_count / samples_with_gt)
        
        self._metrics["teaching_length_mean"].append(sum(teacher_lengths) / len(teacher_lengths))
        self._metrics["teaching_length_std"].append(
            (sum((x - sum(teacher_lengths)/len(teacher_lengths))**2 for x in teacher_lengths) / len(teacher_lengths))**0.5
        )
        
        avg_approach_length = sum(len(self.processing_class.encode(approach)) for approach in student_approaches) / len(student_approaches)
        self._metrics["student_approach_length"].append(avg_approach_length)
        
        efficiency_score = rewards.mean().item() / (sum(teacher_lengths) / len(teacher_lengths) / 100.0)
        self._metrics["token_efficiency"].append(efficiency_score)
        
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / \
            (std_grouped_rewards + 1e-4)
        
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        
        return prompts_text, teacher_completion_ids, advantages
    
    def _extract_solutions_from_responses(self, responses):
        import re
        solutions = []
        for response in responses:
            boxed_matches = re.findall(r'\\boxed\{([^}]+)\}', response)
            if boxed_matches:
                solution_text = boxed_matches[-1].strip()
                solution_text = self._normalize_number(solution_text)
                solutions.append(solution_text)
            else:
                solution_matches = re.findall(r'<solution>(.*?)</solution>', response, re.DOTALL)
                if solution_matches:
                    solution_text = solution_matches[-1].strip()
                    
                    answer_patterns = [
                        r'answer is (?:\\boxed\{)?([^.}\n]+)(?:\})?',
                        r'answer:\s*(?:\\boxed\{)?([^.}\n]+)(?:\})?',
                        r'thus[,\s]+(?:the\s+)?(?:answer\s+is\s+)?([^.}\n]+)',
                        r'therefore[,\s]+(?:the\s+)?(?:answer\s+is\s+)?([^.}\n]+)',
                        r'smallest number of weights needed is (\d+)',
                        r'= (\d+(?:\.\d+)?)\s*(?:grams?|meters?|units?)?'
                    ]
                    
                    found_answer = None
                    for pattern in answer_patterns:
                        matches = re.findall(pattern, solution_text, re.IGNORECASE)
                        if matches:
                            found_answer = matches[-1]
                            break
                    
                    if found_answer:
                        found_answer = self._normalize_number(found_answer)
                        solutions.append(found_answer)
                    else:
                        numbers = re.findall(r'-?\d+(?:\.\d+)?', solution_text)
                        if numbers:
                            solutions.append(self._normalize_number(numbers[-1]))
                        else:
                            solutions.append("")
                else:
                    solutions.append("")
        return solutions
    
    def _normalize_number(self, num_str):
        """Normalize number string for comparison"""
        if not num_str:
            return ""
        num_str = num_str.strip().rstrip('.')
        try:
            num = float(num_str)
            if num == int(num):
                return str(int(num))
            else:
                return f"{num:.2f}"
        except:
            return num_str.strip()
    
    def _check_solution_correctness(self, solution, ground_truth):
        if not solution or not ground_truth:
            return False
        
        solution_normalized = self._normalize_number(solution)
        ground_truth_normalized = self._normalize_number(ground_truth)
        
        if solution_normalized == ground_truth_normalized:
            return True
        
        solution_lower = solution.strip().lower()
        ground_truth_lower = ground_truth.strip().lower()
        
        return solution_lower == ground_truth_lower
    
    def _generate_baseline_solutions(self, prompts_text):
        self._print_debugging_logs('Generating standalone student solutions...')
        
        baseline_prompts = []
        for prompt in prompts_text:
            baseline_prompt = self.student_baseline_template.format(question=prompt)
            baseline_prompts.append(baseline_prompt)
        
        if self.args.use_vllm or self.args.use_ray or self.args.use_vllm_server:
            completion_ids = self._generate_with_vllm(baseline_prompts)
        else:
            completion_ids = self._generate_with_model(self.student_model, baseline_prompts)
        
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True)
        
        return completions_text
    
    def _generate_student_approaches(self, prompts_text):
        probe_prompts = []
        for prompt in prompts_text:
            probe_prompt = self.student_diagnostic_template.format(question=prompt)
            probe_prompts.append(probe_prompt)
        
        if self.args.use_vllm or self.args.use_ray or self.args.use_vllm_server:
            approach_ids = self._generate_with_vllm(probe_prompts, max_tokens=self.max_probe_tokens)
        else:
            approach_ids = self._generate_with_model(self.student_model, probe_prompts, 
                                                    max_tokens=self.max_probe_tokens)
        
        approaches_text = self.processing_class.batch_decode(
            approach_ids, skip_special_tokens=True)
        
        return approaches_text
    
    def _generate_with_model(self, model, prompts_text, max_tokens=None):
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, 
            padding_side="left", add_special_tokens=False
        )
        
        if hasattr(model, 'hf_device_map'):
            device = list(model.hf_device_map.values())[0]
        else:
            device = next(model.parameters()).device
        
        prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]
        
        generation_config = self.generation_config
        if max_tokens is not None:
            generation_config = copy.deepcopy(self.generation_config)
            generation_config.max_new_tokens = max_tokens
        
        with torch.inference_mode():
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, 
                    generation_config=generation_config
                )
        
        prompt_length = prompt_ids.size(1)
        completion_ids = torch.unbind(
            prompt_completion_ids[:, prompt_length:].cpu(), dim=0)
        
        return completion_ids
    
    def _generate_with_vllm(self, prompts_text, max_tokens=None, role="student"):
        if max_tokens is None:
            max_tokens = self.max_completion_length
        
        from accelerate.utils import gather_object
        import concurrent.futures
        
        all_prompts_text = gather_object(prompts_text)
        
        if self.accelerator.is_main_process:
            def generate_with_client(client, prompts):
                return client.generate(
                    prompts=prompts,
                    n=1,
                    repetition_penalty=self.args.repetition_penalty,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    top_k=-1 if self.args.top_k is None else self.args.top_k,
                    min_p=0.0 if self.args.min_p is None else self.args.min_p,
                    max_tokens=max_tokens,
                )
            
            clients = self.teacher_vllm_clients if role == "teacher" else self.student_vllm_clients
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                number_of_prompts = len(all_prompts_text)
                number_of_clients = len(clients)
                base, rem = divmod(number_of_prompts, number_of_clients)
                
                chunk_sizes = [
                    base + (1 if i < rem else 0)
                    for i in range(number_of_clients)
                ]
                
                futures = []
                ss = 0
                for i, client in enumerate(clients):
                    size = chunk_sizes[i]
                    chunk = all_prompts_text[ss:ss + size]
                    futures.append(
                        executor.submit(generate_with_client, client, chunk)
                    )
                    ss += size
                
                completion_ids = [f.result() for f in futures]
                completion_ids = [x for sub in completion_ids for x in sub]
        else:
            completion_ids = [None] * len(all_prompts_text)
        
        completion_ids = broadcast_object_list(
            [completion_ids], from_process=0)[0]
        
        local_batch_size = len(prompts_text)
        process_rank = self.accelerator.process_index
        start_idx = process_rank * local_batch_size
        end_idx = start_idx + local_batch_size
        
        return completion_ids[start_idx:end_idx]
    
