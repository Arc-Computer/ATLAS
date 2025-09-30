import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

import gepa
import yaml
from trainers.prompt_adapter import ATLASGEPAAdapter, ATLASDataInst
from datasets import load_dataset


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(config_path)
    with open(path, 'r') as handle:
        config = yaml.safe_load(handle) or {}

    defaults = config.pop('defaults', [])
    if not defaults:
        return config

    resolved: Dict[str, Any] = {}
    for entry in defaults:
        if isinstance(entry, str):
            default_path = Path(entry)
        elif isinstance(entry, dict):
            key, value = next(iter(entry.items()))
            default_path = Path(key) / value
        else:
            raise ValueError(f"Unsupported defaults entry: {entry}")

        if not default_path.suffix:
            default_path = default_path.with_suffix('.yaml')

        if not default_path.is_absolute():
            default_path = path.parent / default_path

        default_config = load_yaml_config(default_path)
        resolved.update(default_config)

    resolved.update(config)
    return resolved


def load_arc_atlas_dataset_from_hf() -> List[ATLASDataInst]:
    dataset = load_dataset("Arc-Intelligence/Arc-ATLAS-Teach-v1", data_files="curriculum/arc_atlas_teach_rl.jsonl", split="train")
    result = []
    for example in dataset:
        question = (
            example.get("prompt")
            or example.get("problem_text")
            or example.get("question")
            or example.get("input")
            or ""
        )
        ground_truth = (
            example.get("ground_truth")
            or example.get("answer")
            or example.get("solution")
            or ""
        )
        additional_context = {
            k: example[k]
            for k in ["student_approach", "teacher_diagnosis", "teacher_teaching"]
            if k in example
        }
        result.append({
            "question": question,
            "ground_truth": ground_truth,
            "additional_context": additional_context,
        })
        
    return result


def load_gsm8k_zh_dataset() -> List[ATLASDataInst]:
    dataset = load_dataset("meta-math/GSM8K_zh", split="train")
    result = []
    for example in dataset:
        result.append({
            "question": example["question"],
            "ground_truth": example["answer_only"],
            "additional_context": {},
        })
    return result


def load_dynamic_dataset(data_source_config: Dict[str, Any], max_examples: int = 5) -> List[ATLASDataInst]:
    """
    Load dataset dynamically from configured source.
    Supports multiple source types for different agents.
    """
    source_type = data_source_config.get('type', 'file')

    if source_type == 'file':
        return load_dataset_from_jsonl(
            data_source_config['path'],
            data_source_config.get('columns')
        )

    elif source_type == 'http_api':
        import requests
        url = data_source_config['url']
        transform = data_source_config.get('transform', {})

        try:
            response = requests.get(url, timeout=5)
            data = response.json()

            dataset = []
            if transform:
                exec_globals = {'data': data, 'json': json}
                exec(transform.get('script', ''), exec_globals)
                items = exec_globals.get('result', [])
            else:
                items = data if isinstance(data, list) else [data]

            for item in items[:max_examples]:
                dataset.append({
                    "question": item.get(transform.get('question_field', 'question'), ''),
                    "ground_truth": item.get(transform.get('answer_field', 'ground_truth'), '{}'),
                    "additional_context": item.get('context', {}),
                })
            return dataset

        except Exception as e:
            print(f"Failed to fetch from API: {e}")
            return []

    elif source_type == 'prometheus_alerts':
        import requests
        url = data_source_config.get('url', 'http://localhost:9090/api/v1/alerts')

        try:
            response = requests.get(url, timeout=5)
            data = response.json()
            alerts = []

            for alert in data['data']['alerts']:
                if alert['state'] == 'firing':
                    alertname = alert['labels'].get('alertname', 'Unknown')
                    desc = alert['annotations'].get('description', '')
                    alerts.append(f"{alertname}: {desc}")

            if not alerts:
                print("No firing alerts found, using placeholder")
                alerts = ["No active alerts"]

            dataset = []
            for i in range(min(max_examples, max(1, len(alerts) // 5))):
                dataset.append({
                    "question": json.dumps(alerts[i*5:(i+1)*5] if len(alerts) > 5 else alerts),
                    "ground_truth": "{}",
                    "additional_context": {},
                })
            return dataset

        except Exception as e:
            print(f"Failed to fetch Prometheus alerts: {e}")
            return []

    elif source_type == 'itbench_scenarios':
        import yaml
        import glob
        scenarios_dir = data_source_config.get('scenarios_dir')
        if not scenarios_dir:
            raise ValueError("scenarios_dir must be specified in data_source config for itbench_scenarios")

        dataset = []
        spec_files = glob.glob(f"{scenarios_dir}/specs/*.yaml")[:max_examples]

        for spec_file in spec_files:
            incident_id = os.path.basename(spec_file).replace('.yaml', '')
            ground_truth_file = f"{scenarios_dir}/ground_truths/{incident_id}.yaml"

            with open(spec_file, 'r') as f:
                spec = yaml.safe_load(f)

            if os.path.exists(ground_truth_file):
                with open(ground_truth_file, 'r') as f:
                    ground_truth = yaml.safe_load(f)

                expected_json = {
                    "entities": [],
                    "propagations": []
                }

                for group in ground_truth.get('groups', []):
                    expected_json["entities"].append({
                        "id": group['id'],
                        "root_cause": group.get('root_cause', False)
                    })

                for prop in ground_truth.get('propagations', []):
                    expected_json["propagations"].append({
                        "source": prop['source'],
                        "target": prop['target'],
                        "condition": prop.get('condition', ''),
                        "effect": prop.get('effect', '')
                    })

                alert_text = f"Incident: {spec['metadata']['name']}\n"
                alert_text += f"Platform: {spec['metadata']['platform']}\n"
                alert_text += f"Complexity: {spec['metadata']['complexity']}\n"

                if 'faults' in spec['spec']:
                    for fault in spec['spec']['faults']:
                        alert_text += f"Fault configuration: {json.dumps(fault)}\n"

                dataset.append({
                    'question': alert_text,
                    'ground_truth': json.dumps(expected_json),
                    'additional_context': spec
                })

        print(f"Loaded {len(dataset)} ITBench scenarios with ground truth")
        return dataset

    elif source_type == 'custom_function':
        module_path = data_source_config['module']
        function_name = data_source_config['function']

        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_data", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        fetch_function = getattr(module, function_name)
        return fetch_function(max_examples)

    else:
        raise ValueError(f"Unknown data source type: {source_type}")

def load_dataset_from_jsonl(path: str, column_config: Optional[Dict[str, str]] = None) -> List[ATLASDataInst]:
    dataset = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if column_config:
                question_col = column_config.get('question', 'question')
                answer_col = column_config.get('answer', 'ground_truth')
                context_col = column_config.get('context')

                dataset.append({
                    "question": data.get(question_col, ""),
                    "ground_truth": data.get(answer_col, ""),
                    "additional_context": data.get(context_col, {}) if context_col else {},
                })
            else:
                dataset.append({
                    "question": data.get("question", data.get("prompt", "")),
                    "ground_truth": data.get("ground_truth", data.get("answer", "")),
                    "additional_context": data.get("additional_context", {}),
                })
    return dataset



def load_agents_from_config(agent_config):
    if not agent_config:
        return {}
    from wrappers import load_agent
    resolved = {}
    for name, block in agent_config.items():
        if not isinstance(block, dict) or 'provider' not in block:
            raise ValueError(f'Agent {name} is missing provider configuration')
        params = block.get('params', {})
        resolved[name] = load_agent(block['provider'], params)
    return resolved



def run_gepa_optimization(
    teacher_model: Union[str, Callable],
    student_model: Union[str, Callable],
    trainset: List[ATLASDataInst],
    valset: Optional[List[ATLASDataInst]],
    max_metric_calls: int,
    reflection_lm: str,
    trace_storage_path: str,
    seed_prompts: Dict[str, str],
    all_prompts: Dict[str, str],
    gepa_config: Dict[str, Any],
    generation_config: Dict[str, Any],
    wandb_config: Optional[Dict[str, Any]] = None,
    use_vllm_client: bool = False,
    vllm_host: Optional[str] = None,
    vllm_port: Optional[int] = None,
    compatibility_mode: bool = False,
    user_agent: Optional[Callable] = None,
    reflection_instructions: Optional[Dict[str, str]] = None,
    evaluation_config: Optional[Dict[str, Any]] = None,
    optimization_targets: Optional[Dict[str, Any]] = None,
    adapter_instance: Optional[Any] = None,
) -> Dict:

    if adapter_instance:
        adapter = adapter_instance
    elif compatibility_mode and user_agent:
        from trainers.compatibility_adapter import CompatibilityAdapter
        adapter = CompatibilityAdapter(
            teacher_model=teacher_model,
            user_agent=user_agent,
            trace_storage_path=trace_storage_path,
            generation_config=generation_config,
            max_litellm_workers=generation_config.get('max_litellm_workers', 10),
            reflection_instructions=reflection_instructions,
            evaluation_config=evaluation_config,
            optimization_targets=optimization_targets,
            student_model=student_model,
        )
        adapter.total_evaluations = max_metric_calls
    else:
        adapter = ATLASGEPAAdapter(
            teacher_model=teacher_model,
            student_model=student_model,
            reward_function=None,
            trace_storage_path=trace_storage_path,
            all_prompts=all_prompts,
            generation_config=generation_config,
            max_litellm_workers=generation_config.get('max_litellm_workers', 10),
            use_vllm_client=use_vllm_client,
            vllm_host=vllm_host,
            vllm_port=vllm_port,
        )

    import litellm
    def reflection_lm_func(prompt: str) -> str:
        try:
            if adapter and hasattr(adapter, 'display_manager') and adapter.display_manager:
                from trainers.terminal_display import TerminalDisplay
                display = adapter.display_manager.display
                if display and isinstance(display, TerminalDisplay):
                    print(f"\n💭 Reflection model analyzing performance...")

            response = litellm.completion(
                model=reflection_lm,
                messages=[{"role": "user", "content": prompt}],
                timeout=generation_config.get('timeout', 600),
                request_timeout=generation_config.get('request_timeout', 600),
                max_tokens=generation_config.get('reflection_max_tokens', 32768),
                temperature=generation_config.get('temperature', 0.7)
            )
            if response and response.choices and len(response.choices) > 0 and response.choices[0].message:
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("Reflection LM returned None content")

                if adapter and hasattr(adapter, 'display_manager') and adapter.display_manager:
                    display = adapter.display_manager.display
                    if display and isinstance(display, TerminalDisplay):
                        display._print_box("REFLECTION RESPONSE", content)

                return content
            else:
                raise ValueError("Invalid response structure from reflection LM")

        except Exception as e:
            print(f"[ERROR] Reflection LM failed: {e} (Model: {reflection_lm})")
            raise


    result = gepa.optimize(
        seed_candidate=seed_prompts,
        trainset=trainset,
        valset=valset if valset else trainset,
        adapter=adapter,
        reflection_lm=reflection_lm_func,
        max_metric_calls=max_metric_calls,
        **gepa_config
    )

    print("\nGEPA optimization completed!")
    
    return result


def save_optimized_prompts(result, output_path: str, initial_score: float = None):
    output_data = {
        "best_candidate": result.best_candidate,
        "best_score": float(result.best_score) if hasattr(result, 'best_score') else None,
        "initial_score": initial_score,
        "improvement": float(result.best_score - initial_score) if initial_score and hasattr(result, 'best_score') else None,
        "improvement_percentage": float((result.best_score - initial_score) / initial_score * 100) if initial_score and hasattr(result, 'best_score') and initial_score > 0 else None,
        "pareto_frontier": result.pareto_frontier if hasattr(result, 'pareto_frontier') else None,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nOptimized prompts saved to: {output_path}")
    if output_data.get('best_score') is not None:
        print(f"Best score achieved: {output_data['best_score']:.4f}")
    if output_data.get('initial_score') is not None:
        print(f"Initial score: {output_data['initial_score']:.4f}")
    if output_data.get('improvement_percentage') is not None:
        print(f"Performance gain: {output_data['improvement_percentage']:.2f}%")


def main():
    import logging
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.ERROR)
    os.environ["LITELLM_LOG"] = "ERROR"

    parser = argparse.ArgumentParser(description="Optimize ATLAS teaching prompts using reflective evolution")
    
    parser.add_argument(
        "--trainset",
        type=str,
        required=False,
        default=None,
        help="Path to training dataset (JSONL format)",
    )
    parser.add_argument(
        "--valset",
        type=str,
        default=None,
        help="Path to validation dataset (JSONL format)",
    )
    parser.add_argument(
        "--student-model",
        type=str,
        required=False,
        default=None,
        help="Student model",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        required=False,
        default=None,
        help="Teacher model",
    )
    parser.add_argument(
        "--reflection-lm",
        type=str,
        default=None,
        help="Language model for GEPA reflection (defaults to teacher model)",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=150,
        help="Maximum number of metric evaluations",
    )
    parser.add_argument(
        "--trace-storage",
        type=str,
        default="traces/gepa_traces.jsonl",
        help="Path to store execution traces",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="optimized_prompts.json",
        help="Path to save optimized prompts",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/optimize/default.yaml",
        help="Path to configuration file with seed prompts",
    )
    parser.add_argument(
        "--use-vllm-client",
        action="store_true",
        help="Use vLLM client for generation",
    )
    parser.add_argument(
        "--vllm-host",
        type=str,
        default=None,
        help="vLLM server host (required if --use-vllm-client)",
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=None,
        help="vLLM server port (required if --use-vllm-client)",
    )

    args = parser.parse_args()
    
    if args.use_vllm_client and (not args.vllm_host or not args.vllm_port):
        parser.error("--vllm-host and --vllm-port are required when using --use-vllm-client")
    
    print("Loading configuration...")
    import yaml
    from pathlib import Path

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if 'defaults' in config:
        base_configs = config.get('defaults', [])
        base_config = {}
        config_dir = Path(args.config).parent

        for base_name in base_configs:
            base_path = config_dir / f"{base_name}.yaml"
            if base_path.exists():
                with open(base_path, 'r') as f:
                    base = yaml.safe_load(f)
                    base_config.update(base)

        config_without_defaults = {k: v for k, v in config.items() if k != 'defaults'}
        base_config.update(config_without_defaults)
        config = base_config
    
    compatibility_mode = config.get('compatibility_mode', False)

    reflection_lm = args.reflection_lm or config.get('reflection_lm') or config.get('teacher_model') or args.teacher_model

    seed_prompts = config.get('seed_prompts', {})
    if not seed_prompts and not compatibility_mode:
        parser.error(f"No seed_prompts found in config file: {args.config}")
    
    fixed_prompts = config.get('fixed_prompts', {})
    all_prompts = {**seed_prompts, **fixed_prompts}
    
    gepa_config = config.get('gepa_config', {})
    wandb_config = config.get('wandb', {})
    generation_config = config.get('generation_config', {})
    generation_config['max_litellm_workers'] = config.get('max_litellm_workers', 10)
    reflection_instructions = config.get('reflection_instructions') or {}
    evaluation_config = config.get('evaluation') or {}
    optimization_targets = config.get('optimization_targets') or {}
    
    print("Loading datasets...")
    data_config = config.get('data', {})
    column_config = data_config.get('columns')
    max_examples = config.get('max_examples', 5)

    data_source = config.get('data_source')
    if data_source:
        print(f"Using dynamic data source: {data_source.get('type')}")
        trainset = load_dynamic_dataset(data_source, max_examples)
        valset = trainset
    elif args.trainset and args.trainset == "arc-atlas-rl":
        trainset = load_arc_atlas_dataset_from_hf()
        valset = None
    elif args.trainset and args.trainset == "gsm8k-zh":
        trainset = load_gsm8k_zh_dataset()
        valset = None
    elif args.trainset:
        trainset = load_dataset_from_jsonl(args.trainset, column_config)
        valset = load_dataset_from_jsonl(args.valset, column_config) if args.valset else None
    else:
        raise ValueError("No data source specified. Either provide --trainset or configure data_source in config file")

    if max_examples and not data_source:
        trainset = trainset[:max_examples]
        if valset:
            valset = valset[:max_examples]
        print(f"Limited dataset to {max_examples} examples")

    print(f"Loaded {len(trainset)} training examples")
    
    if valset:
        print(f"Loaded {len(valset)} validation examples")
    
    agent_handles = load_agents_from_config(config.get('agents'))
    
    teacher_model = agent_handles.get('teacher')
    if teacher_model is None:
        if config.get('teacher_wrapper'):
            from wrappers import load_wrapper
            teacher_model = load_wrapper(
                config['teacher_wrapper']['type'],
                config['teacher_wrapper']['config']
            )
        else:
            teacher_model = args.teacher_model

    student_model = agent_handles.get('student')
    if student_model is None:
        if config.get('student_wrapper'):
            from wrappers import load_wrapper
            student_model = load_wrapper(
                config['student_wrapper']['type'],
                config['student_wrapper']['config']
            )
        else:
            student_model = args.student_model

    user_agent = agent_handles.get('target') or agent_handles.get('user_agent')

    if compatibility_mode:
        import sys
        import io
        from trainers.terminal_display import DisplayManager

        if not user_agent:
            agent_type = config.get('agent_type')
            agent_config = config.get('agent_config', {})
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            try:
                if agent_type:
                    from wrappers import load_wrapper
                    user_agent = load_wrapper(agent_type, agent_config)
                elif config.get('user_agent'):
                    from wrappers import load_wrapper
                    user_agent = load_wrapper(
                        config['user_agent']['type'],
                        config['user_agent']['config']
                    )
                else:
                    raise ValueError("Compatibility mode requires agent configuration")
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        if not user_agent:
            raise ValueError('Compatibility mode requires agent configuration')

        display_manager = DisplayManager(verbose=True)
        display_manager.start(args.max_metric_calls)
        result_container = {'result': None, 'error': None}

        if not seed_prompts:
            seed_prompts_local = {
                "teacher_adaptive_template":
                    "You are an expert teacher. The student gave this response: {baseline_response}\n\n"
                    "To the question: {question}\n\n"
                    "Provide focused teaching to help them improve. Wrap teaching in <teaching> tags."
            }
        else:
            seed_prompts_local = seed_prompts

        from trainers.compatibility_adapter import CompatibilityAdapter
        adapter_instance = CompatibilityAdapter(
            teacher_model=teacher_model,
            user_agent=user_agent,
            trace_storage_path=args.trace_storage,
            generation_config=generation_config,
            reflection_instructions=reflection_instructions,
            evaluation_config=evaluation_config,
            optimization_targets=optimization_targets,
            student_model=student_model,
        )
        adapter_instance.total_evaluations = args.max_metric_calls
        adapter_instance.display_manager = display_manager

        result_container['result'] = run_gepa_optimization(
            teacher_model=teacher_model,
            student_model=student_model,
            trainset=trainset,
            valset=valset,
            max_metric_calls=args.max_metric_calls,
            reflection_lm=reflection_lm,
            trace_storage_path=args.trace_storage,
            seed_prompts=seed_prompts_local,
            all_prompts=all_prompts,
            gepa_config=gepa_config,
            generation_config=generation_config,
            wandb_config=wandb_config,
            use_vllm_client=args.use_vllm_client,
            vllm_host=args.vllm_host,
            vllm_port=args.vllm_port,
            compatibility_mode=compatibility_mode,
            user_agent=user_agent,
            reflection_instructions=reflection_instructions,
            evaluation_config=evaluation_config,
            optimization_targets=optimization_targets,
            adapter_instance=adapter_instance,
        )

        display_manager.stop()

        if result_container['error']:
            raise result_container['error']

        result = result_container['result']
    else:
        result = run_gepa_optimization(
            teacher_model=teacher_model,
            student_model=student_model,
            trainset=trainset,
            valset=valset,
            max_metric_calls=args.max_metric_calls,
            reflection_lm=reflection_lm,
            trace_storage_path=args.trace_storage,
            seed_prompts=seed_prompts,
            all_prompts=all_prompts,
            gepa_config=gepa_config,
            generation_config=generation_config,
            wandb_config=wandb_config,
            use_vllm_client=args.use_vllm_client,
            vllm_host=args.vllm_host,
            vllm_port=args.vllm_port,
            compatibility_mode=compatibility_mode,
            user_agent=user_agent,
            reflection_instructions=reflection_instructions,
            evaluation_config=evaluation_config,
            optimization_targets=optimization_targets,
        )

    if result:
        initial_score = result.val_aggregate_scores[0] if hasattr(result, 'val_aggregate_scores') and result.val_aggregate_scores else None
        save_optimized_prompts(result, args.output, initial_score=initial_score)

        from trainers.terminal_display import TerminalDisplay
        display = TerminalDisplay(verbose=True)

        print("\n" + "="*80)
        print("🎯 OPTIMIZATION COMPLETE - FINAL OPTIMIZED TEMPLATES")
        print("="*80)

        if hasattr(result, 'best_candidate') and result.best_candidate:
            for key, value in result.best_candidate.items():
                title = key.replace('_', ' ').upper()
                display._print_box(title, value)
    else:
        print("No optimization result returned")


if __name__ == "__main__":
    main()