import re
import ast
import operator
import json
from typing import Any, Dict, List, Optional, Callable
from .extraction_utils import ATLASExtractionUtils


class ConfigurableEvaluator:
    """User-configurable evaluation system for ATLAS optimization."""

    def __init__(self, evaluation_config: Optional[Dict[str, Any]] = None):
        self.config = evaluation_config or {}
        self.metrics_config = self.config.get('metrics', [])
        self.reward_formula = self.config.get('reward_formula')
        self.custom_functions = self.config.get('custom_functions', {})
        self._compiled_patterns = {}
        self.last_metrics = {}
        self.validate_config()

    def validate_config(self):
        """Validate evaluation configuration."""
        if not self.metrics_config:
            self.metrics_config = [
                {'name': 'correctness', 'type': 'exact_match', 'weight': 0.7},
                {'name': 'efficiency', 'type': 'token_reduction', 'weight': 0.3}
            ]

        total_weight = sum(m.get('weight', 0) for m in self.metrics_config)
        if abs(total_weight - 1.0) > 0.01:
            print(f"Warning: Metric weights sum to {total_weight}, normalizing to 1.0")
            if total_weight > 0:
                for metric in self.metrics_config:
                    metric['weight'] = metric.get('weight', 0) / total_weight

    def calculate_metric(
        self,
        metric_config: Dict[str, Any],
        response: str,
        baseline: str,
        ground_truth: str,
        question: str
    ) -> float:
        """Calculate a single metric based on configuration."""
        metric_type = metric_config.get('type', 'exact_match')

        if metric_type == 'exact_match':
            extracted = ATLASExtractionUtils.extract_solution(response)
            return 1.0 if ATLASExtractionUtils.check_correctness(extracted, ground_truth) else 0.0

        elif metric_type == 'contains':
            target = metric_config.get('target', ground_truth)
            return 1.0 if target.lower() in response.lower() else 0.0

        elif metric_type == 'regex':
            pattern = metric_config.get('pattern')
            if pattern:
                try:
                    compiled_pattern = re.compile(pattern)
                    return 1.0 if compiled_pattern.search(response) else 0.0
                except re.error:
                    return 0.0
            return 0.0

        elif metric_type == 'token_reduction':
            baseline_tokens = len(baseline.split())
            response_tokens = len(response.split())
            if baseline_tokens > 0 and response_tokens < baseline_tokens:
                return (baseline_tokens - response_tokens) / baseline_tokens
            return 0.0

        elif metric_type == 'length_penalty':
            max_length = metric_config.get('max_length', 500)
            response_length = len(response.split())
            if response_length <= max_length:
                return 1.0
            else:
                penalty = (response_length - max_length) / max_length
                return max(0, 1.0 - penalty)

        elif metric_type == 'json_comparison':
            return self._compare_json_output(response, ground_truth, metric_config)

        elif metric_type == 'custom':
            function_name = metric_config.get('function')
            if function_name in self.custom_functions:
                try:
                    custom_func = self.custom_functions[function_name]
                    return custom_func(response, baseline, ground_truth, question)
                except Exception as e:
                    print(f"Custom function {function_name} failed: {e}")
                    return 0.0

        return 0.0

    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """Extract JSON from response text."""
        try:
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
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

        # If no JSON found and response looks like CrewAI output, try to parse it
        if response and isinstance(response, str):
            if "Fault Propagation Chain:" in response or "- Entity:" in response or "- Alert:" in response:
                return self._parse_crewai_text_to_json(response)

        return None

    def _convert_to_topology_ids(self, ground_truth: Dict) -> Dict:
        """Convert ground truth to use topology hash IDs like the agent does."""
        import os

        # Load topology mapping
        topology_path = "/Users/arc-aman/Documents/GitHub/ATLAS/ITBench-SRE-Agent/src/lumyn/tools/report_generation/data/topology_nodes.json"
        if not os.path.exists(topology_path):
            print(f"WARNING: Topology file not found at {topology_path}")
            return ground_truth

        try:
            with open(topology_path, 'r') as f:
                topology = json.load(f)

            # Build name to ID mapping
            name_to_id = {}
            for node in topology:
                if 'name' in node and 'id' in node:
                    name = node['name']
                    name_to_id[name] = node['id']

                    # Map variants of the name
                    if name.startswith('otel-demo-'):
                        # Map without otel-demo prefix
                        short_name = name[10:]  # Remove 'otel-demo-'
                        name_to_id[short_name] = node['id']

                        # Map service names
                        if 'service' in short_name:
                            service_name = short_name.replace('service', '')
                            name_to_id[service_name] = node['id']

            # Convert entities
            if 'entities' in ground_truth:
                converted_entities = []
                for entity in ground_truth['entities']:
                    entity_id = entity.get('id', '')

                    # Try to find matching topology ID
                    possible_names = [
                        entity_id,
                        f"otel-demo-{entity_id}",
                        entity_id.replace('-1', ''),  # Remove -1 suffix
                        entity_id.replace('_', '-'),
                        entity_id.replace('-pod', ''),
                        entity_id.replace('-service', 'service'),
                        # Special case for load-generator
                        'otel-demo-loadgenerator' if 'load-generator' in entity_id else None,
                        # Special case for frontend-proxy
                        'otel-demo-frontendproxy' if 'frontend-proxy' in entity_id else None,
                    ]

                    mapped_id = None
                    for name in possible_names:
                        if name and name in name_to_id:
                            mapped_id = name_to_id[name]
                            print(f"Mapped ground truth entity: {entity_id} -> {mapped_id}")
                            break

                    if mapped_id:
                        entity['id'] = mapped_id
                    else:
                        print(f"WARNING: No topology mapping found for entity: {entity_id}")

                    converted_entities.append(entity)

                ground_truth['entities'] = converted_entities

            # Convert propagations
            if 'propagations' in ground_truth:
                for prop in ground_truth['propagations']:
                    # Map source and target IDs
                    for field in ['source', 'target']:
                        if field in prop:
                            orig_id = prop[field]
                            possible_names = [
                                orig_id,
                                f"otel-demo-{orig_id}",
                                orig_id.replace('-1', ''),
                                orig_id.replace('_', '-'),
                                orig_id.replace('-pod', ''),
                                orig_id.replace('-service', 'service'),
                                'otel-demo-loadgenerator' if 'load-generator' in orig_id else None,
                                'otel-demo-frontendproxy' if 'frontend-proxy' in orig_id else None,
                            ]

                            for name in possible_names:
                                if name and name in name_to_id:
                                    prop[field] = name_to_id[name]
                                    print(f"Mapped propagation {field}: {orig_id} -> {name_to_id[name]}")
                                    break

            return ground_truth

        except Exception as e:
            print(f"ERROR converting to topology IDs: {e}")
            return ground_truth

    def _parse_crewai_text_to_json(self, text: str) -> Optional[Dict]:
        """Parse CrewAI text output into expected JSON format."""
        result = {
            "entities": [],
            "propagations": []
        }

        lines = text.strip().split('\n')
        entities_list = []

        for line in lines:
            line = line.strip()
            # Extract entities from lines like "- Entity: product-catalog Deployment (otel-demo namespace)"
            if '- Entity:' in line or 'Entity:' in line:
                entity_text = line.replace('- Entity:', '').replace('Entity:', '').strip()
                # Clean up the entity ID
                if entity_text and entity_text not in [e for e in entities_list]:
                    entities_list.append(entity_text)

        # First entity is typically the root cause
        for i, entity in enumerate(entities_list):
            result["entities"].append({
                "id": entity,
                "root_cause": i == 0
            })

            # Create propagation chain
            if i > 0:
                result["propagations"].append({
                    "source": entities_list[i-1],
                    "target": entity,
                    "condition": "Service dependency",
                    "effect": "Cascading failure"
                })

        # Return None if no entities found
        if not result["entities"]:
            return None

        return result

    def _itbench_sre_evaluation(self, agent_output: Dict, ground_truth: Dict) -> float:
        """ITBench-specific SRE evaluation with partial credit."""
        score = 0.0

        # 1. Root cause identification (40% weight)
        gt_root_causes = [e for e in ground_truth.get('entities', []) if e.get('root_cause', False)]
        agent_root_causes = [e for e in agent_output.get('entities', []) if e.get('root_cause', False)]

        if gt_root_causes and agent_root_causes:
            # Full credit if any root cause identified
            score += 0.4
        elif agent_output.get('entities'):
            # Check if agent identified relevant entities even without marking root cause
            for agent_entity in agent_output['entities']:
                agent_id = agent_entity.get('id', '').lower()
                # Check for product-catalog or productcatalog mentions
                if 'product' in agent_id and 'catalog' in agent_id:
                    score += 0.2  # Partial credit for identifying relevant entity
                    break

        # 2. Entity coverage (30% weight)
        gt_entity_count = len(ground_truth.get('entities', []))
        agent_entity_count = len(agent_output.get('entities', []))
        if gt_entity_count > 0:
            entity_coverage = min(agent_entity_count / gt_entity_count, 1.0)
            score += 0.3 * entity_coverage

        # 3. Propagation identification (30% weight)
        gt_prop_count = len(ground_truth.get('propagations', []))
        agent_prop_count = len(agent_output.get('propagations', []))
        if gt_prop_count > 0:
            prop_coverage = min(agent_prop_count / gt_prop_count, 1.0)
            score += 0.3 * prop_coverage
        elif agent_prop_count > 0:
            # Some credit for identifying any propagations
            score += 0.1

        return score

    def _compare_json_output(self, response: str, ground_truth_str: str, metric_config: Dict) -> float:
        """Generic JSON comparison based on configuration."""
        try:
            ground_truth = json.loads(ground_truth_str) if isinstance(ground_truth_str, str) else ground_truth_str
        except Exception as e:
            print(f"ERROR: Failed to parse ground truth JSON: {e}")
            print(f"Ground truth string: {ground_truth_str[:200] if ground_truth_str else 'None'}")
            return 0.0

        if not isinstance(ground_truth, dict):
            print(f"ERROR: Ground truth is not a dict, got type: {type(ground_truth)}")
            return 0.0

        agent_output = self._extract_json_from_response(response)
        if agent_output is None:
            print(f"WARNING: Could not extract JSON from response")
            print(f"Response: {response[:200] if response else 'Empty response'}")
            return 0.0

        if not isinstance(agent_output, dict):
            print(f"ERROR: Extracted output is not a dict, got type: {type(agent_output)}")
            print(f"Extracted value: {agent_output}")
            return 0.0

        # Use ITBench-specific evaluation if configured
        if metric_config.get('use_itbench_evaluation', False):
            return self._itbench_sre_evaluation(agent_output, ground_truth)

        # For ITBench SRE: Convert ground truth to use topology hash IDs if needed
        if metric_config.get('use_topology_mapping', False):
            ground_truth = self._convert_to_topology_ids(ground_truth)

        comparison_fields = metric_config.get('comparison_fields', {})
        if not comparison_fields:
            return 1.0 if agent_output == ground_truth else 0.0

        total_score = 0.0
        total_weight = 0.0

        for field, field_config in comparison_fields.items():
            weight = field_config.get('weight', 1.0)
            mode = field_config.get('mode', 'exact')

            try:
                gt_value = ground_truth.get(field)
                agent_value = agent_output.get(field)

                if gt_value is None:
                    print(f"Field '{field}' not found in ground truth")
                    continue

                if agent_value is None:
                    print(f"Field '{field}' not found in agent output")
                    continue

                field_score = 0.0

                if mode == 'exact':
                    field_score = 1.0 if gt_value == agent_value else 0.0
                elif mode == 'recursive':
                    field_score = self._recursive_compare(gt_value, agent_value, field_config)
                elif mode == 'list_match':
                    field_score = self._list_compare(gt_value, agent_value, field_config)

                total_score += field_score * weight
                total_weight += weight

            except Exception as e:
                print(f"ERROR comparing field '{field}': {e}")
                continue

        return total_score / total_weight if total_weight > 0 else 0.0

    def _recursive_compare(self, gt_value, agent_value, config):
        """Recursively compare nested structures."""
        if type(gt_value) != type(agent_value):
            return 0.0

        if isinstance(gt_value, dict):
            matching_keys = set(gt_value.keys()) & set(agent_value.keys())
            if not matching_keys:
                return 0.0
            scores = [self._recursive_compare(gt_value[k], agent_value[k], config) for k in matching_keys]
            return sum(scores) / len(gt_value) if gt_value else 0.0

        elif isinstance(gt_value, list):
            return self._list_compare(gt_value, agent_value, config)
        else:
            return 1.0 if gt_value == agent_value else 0.0

    def _list_compare(self, gt_list, agent_list, config):
        """Compare lists based on configured strategy."""
        strategy = config.get('list_strategy', 'exact')

        if strategy == 'exact':
            return 1.0 if gt_list == agent_list else 0.0
        elif strategy == 'contains_all':
            if not gt_list:
                return 1.0
            matches = sum(1 for item in gt_list if item in agent_list)
            return matches / len(gt_list)
        elif strategy == 'match_by_key':
            key_field = config.get('key_field', 'id')
            gt_keyed = {item.get(key_field): item for item in gt_list if isinstance(item, dict)}
            agent_keyed = {item.get(key_field): item for item in agent_list if isinstance(item, dict)}

            if not gt_keyed:
                return 0.0

            score = 0.0
            for key, gt_item in gt_keyed.items():
                if key in agent_keyed:
                    item_score = self._recursive_compare(gt_item, agent_keyed[key], config)
                    score += item_score

            return score / len(gt_keyed)

        return 0.0

    def evaluate(
        self,
        response: str,
        baseline: str,
        ground_truth: str,
        question: str
    ) -> Dict[str, float]:
        """Evaluate response using all configured metrics."""
        metrics = {}

        for metric_config in self.metrics_config:
            metric_name = metric_config.get('name', 'unnamed')
            metric_value = self.calculate_metric(
                metric_config, response, baseline, ground_truth, question
            )
            metrics[metric_name] = metric_value

        return metrics

    def _safe_eval_formula(self, formula: str, variables: Dict[str, Any]) -> float:
        """Safely evaluate a mathematical formula with restricted operations."""
        safe_dict = {
            'metrics': variables.get('metrics', {}),
            'baseline_metrics': variables.get('baseline_metrics', {}),
            '__builtins__': {}
        }

        try:
            exec(compile(formula, '<string>', 'exec'), safe_dict)

            for var_name in safe_dict:
                if var_name not in ['metrics', 'baseline_metrics', '__builtins__'] and isinstance(safe_dict[var_name], (int, float)):
                    return float(safe_dict[var_name])

            return 0.0
        except Exception:
            try:
                tree = ast.parse(formula, mode='eval')

                def _eval_node(node):
                    if isinstance(node, ast.Constant):
                        return node.value
                    elif isinstance(node, ast.Name):
                        if node.id in safe_dict:
                            return safe_dict[node.id]
                        raise ValueError(f"Name '{node.id}' not allowed")
                    elif isinstance(node, ast.BinOp):
                        allowed_ops = {
                            ast.Add: operator.add,
                            ast.Sub: operator.sub,
                            ast.Mult: operator.mul,
                            ast.Div: operator.truediv,
                            ast.Mod: operator.mod,
                            ast.Pow: operator.pow,
                        }
                        op = type(node.op)
                        if op in allowed_ops:
                            left = _eval_node(node.left)
                            right = _eval_node(node.right)
                            return allowed_ops[op](left, right)
                        raise ValueError(f"Operation {op} not allowed")
                    elif isinstance(node, ast.Subscript):
                        value = _eval_node(node.value)
                        if isinstance(node.slice, ast.Constant):
                            key = node.slice.value
                        elif isinstance(node.slice, ast.Index):
                            key = _eval_node(node.slice.value)
                        else:
                            raise ValueError("Complex subscript not allowed")
                        return value.get(key, 0.0) if isinstance(value, dict) else 0.0
                    elif isinstance(node, ast.IfExp):
                        test = _eval_node(node.test)
                        if test:
                            return _eval_node(node.body)
                        else:
                            return _eval_node(node.orelse)
                    elif isinstance(node, ast.Compare):
                        allowed_compare_ops = {
                            ast.Gt: operator.gt,
                            ast.Lt: operator.lt,
                            ast.GtE: operator.ge,
                            ast.LtE: operator.le,
                            ast.Eq: operator.eq,
                            ast.NotEq: operator.ne,
                        }
                        left = _eval_node(node.left)
                        for op, comparator in zip(node.ops, node.comparators):
                            right = _eval_node(comparator)
                            op_type = type(op)
                            if op_type in allowed_compare_ops:
                                if not allowed_compare_ops[op_type](left, right):
                                    return False
                                left = right
                            else:
                                raise ValueError(f"Comparison {op_type} not allowed")
                        return True
                    else:
                        raise ValueError(f"Node type {type(node)} not allowed")

                return float(_eval_node(tree.body))
            except Exception:
                return 0.0

    def calculate_reward(
        self,
        metrics: Dict[str, float],
        baseline_metrics: Dict[str, float]
    ) -> float:
        """Calculate final reward using configured formula or default logic."""

        if self.reward_formula:
            result = self._safe_eval_formula(self.reward_formula, {
                'metrics': metrics,
                'baseline_metrics': baseline_metrics
            })
            if result != 0.0:
                return result

        # Default reward logic (backward compatible)
        weighted_score = 0.0
        for metric_config in self.metrics_config:
            metric_name = metric_config.get('name')
            weight = metric_config.get('weight', 0)

            if metric_name in metrics:
                improvement = metrics[metric_name] - baseline_metrics.get(metric_name, 0)
                weighted_score += improvement * weight

        return weighted_score

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> List[float]:
        """Compatible interface with OnlineTeachingReward."""
        rewards = []
        ground_truths = kwargs.get('ground_truths', [])
        baseline_solutions = kwargs.get('baseline_solutions', [])
        solutions = kwargs.get('solutions', [])
        questions = kwargs.get('questions', prompts)

        for i in range(len(prompts)):
            ground_truth = ground_truths[i] if i < len(ground_truths) else ""
            baseline = baseline_solutions[i] if i < len(baseline_solutions) else ""
            solution = solutions[i] if i < len(solutions) else ""
            question = questions[i] if i < len(questions) else ""

            if not ground_truth:
                rewards.append(0.0)
                continue

            solution_metrics = self.evaluate(solution, baseline, ground_truth, question)
            baseline_metrics = self.evaluate(baseline, baseline, ground_truth, question)

            self.last_metrics = solution_metrics

            reward = self.calculate_reward(solution_metrics, baseline_metrics)
            rewards.append(reward)

        return rewards