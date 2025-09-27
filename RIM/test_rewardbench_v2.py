import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datasets import load_dataset
import numpy as np
from typing import Dict, List, Any
from RIM.judges import AccuracyJudge
from RIM.model_interface import model_interface
import json
import yaml
import os
import logging
from datetime import datetime


def evaluate_pairwise_accuracy(trajectory: Dict[str, str], judge: AccuracyJudge, rim_config: Dict[str, Any]) -> Dict[str, Any]:
    temperatures: List[float] = rim_config['temperatures']
    variance_threshold: float = rim_config['variance_threshold']
    small_model: str = rim_config['models']['small_model']
    large_model: str = rim_config['models']['large_model']
    default_max_tokens = rim_config.get('model_configs', {}).get('default', {}).get('max_tokens', 32768)
    large_max_tokens = rim_config.get('model_configs', {}).get('large', {}).get('max_tokens', 32768)

    samples: List[Dict[str, Any]] = []

    for temperature in temperatures:
        def model_fn(prompt: str, temp: float) -> str:
            return model_interface.call_model(
                model_name=small_model,
                prompt=prompt,
                temperature=temp,
                max_tokens=default_max_tokens
            )

        result = judge.evaluate(trajectory, model_fn, temperature)
        result['temperature'] = temperature
        samples.append(result)

    if not samples:
        return {
            'score_a': 0.0,
            'score_b': 0.0,
            'explanation': 'No valid evaluations produced',
            'uncertainty': 1.0,
            'principles': []
        }

    score_diffs = [sample.get('score_a', 0.0) - sample.get('score_b', 0.0) for sample in samples]
    variance = np.std(score_diffs)
    max_uncertainty = max(sample.get('uncertainty', 0.5) for sample in samples)

    if variance <= variance_threshold and max_uncertainty <= 0.3:
        return min(samples, key=lambda s: s.get('uncertainty', 0.5))

    meta_prompt = build_pairwise_meta_prompt(trajectory, samples)
    response = model_interface.call_model(
        model_name=large_model,
        prompt=meta_prompt,
        temperature=0.3,
        max_tokens=large_max_tokens
    )

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        parsed = {
            'score_a': 0.0,
            'score_b': 0.0,
            'explanation': 'Failed to parse large judge response',
            'uncertainty': 1.0,
            'principles': []
        }

    return {
        'score_a': parsed.get('score_a', 0.0),
        'score_b': parsed.get('score_b', 0.0),
        'explanation': parsed.get('explanation', parsed.get('rationale', '')),
        'uncertainty': parsed.get('uncertainty', 0.5),
        'principles': parsed.get('principles', [])
    }


def build_pairwise_meta_prompt(trajectory: Dict[str, str], samples: List[Dict[str, Any]]) -> str:
    samples_text = "\n\n".join([
        (
            f"Evaluation {idx + 1} (temperature {sample.get('temperature', 0.0):.2f}):\n"
            f"Principles: {json.dumps(sample.get('principles', []))}\n"
            f"Scores -> Response A: {sample.get('score_a', 0.0):.2f}, Response B: {sample.get('score_b', 0.0):.2f}\n"
            f"Uncertainty: {sample.get('uncertainty', 0.5):.2f}\n"
            f"Explanation: {sample.get('explanation', '')}"
        )
        for idx, sample in enumerate(samples)
    ])

    return f"""Multiple principle-based judges evaluated two candidate responses but reached disagreement or high uncertainty. Consolidate their reasoning and produce a final decision.

Prompt: {trajectory.get('prompt', '')}

Response A: {trajectory.get('response_a', '')}

Response B: {trajectory.get('response_b', '')}

Previous evaluations:
{samples_text}

Instructions:
1. Identify the most reliable principles across the evaluations or draft improved principles if necessary.
2. Compare both responses against those principles.
3. Explain clearly which response is preferred and why.
4. Provide calibrated uncertainty in [0.0, 1.0]; values >0.3 indicate remaining doubt.

Output JSON only: {{"principles": [{{"name": str, "weight": float, "description": str}}], "score_a": float, "score_b": float, "explanation": str, "uncertainty": float}}"""


def test_rim_on_rewardbench(num_samples: int = 100):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'RIM/benchmark_run_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("=== Starting RewardBench Evaluation ===")

    dataset = load_dataset("allenai/reward-bench-2", split="test")
    logger.info(f"Loaded dataset: {len(dataset)} total samples")
    logger.info(f"Testing on {min(num_samples, len(dataset))} samples from test split")

    with open('configs/rim_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    accuracy_judge = AccuracyJudge()
    rim_config = config['rim']

    correct = 0
    total = 0
    results = []
    subset_stats = {}
    subset_results = {}

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        subset = sample['subset']

        logger.info(f"Processing sample {i+1}/{min(num_samples, len(dataset))}, Subset: {subset}, ID: {sample['id']}")

        trajectory = {
            'prompt': sample['prompt'],
            'response_a': sample['chosen'],
            'response_b': sample['rejected']
        }

        try:
            result = evaluate_pairwise_accuracy(
                trajectory=trajectory,
                judge=accuracy_judge,
                rim_config=rim_config
            )

            score_chosen = result['score_a']
            score_rejected = result['score_b']

            is_correct = score_chosen > score_rejected
            if is_correct:
                correct += 1
            total += 1

            if subset not in subset_stats:
                subset_stats[subset] = {'correct': 0, 'total': 0}
                subset_results[subset] = []
            subset_stats[subset]['total'] += 1
            if is_correct:
                subset_stats[subset]['correct'] += 1

            result_entry = {
                'sample_id': sample['id'],
                'subset': sample['subset'],
                'score_chosen': score_chosen,
                'score_rejected': score_rejected,
                'explanation': result['explanation'],
                'correct': is_correct
            }
            results.append(result_entry)
            subset_results[subset].append(result_entry)

            logger.info(f"Sample {i+1} - Scores: chosen={score_chosen:.2f}, rejected={score_rejected:.2f}, correct={is_correct}")

            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{min(num_samples, len(dataset))}, Overall Accuracy: {correct/total:.3f}")

            prev_subset = dataset[i-1]['subset'] if i > 0 else None
            next_subset = dataset[i+1]['subset'] if i < min(num_samples, len(dataset)) - 1 else None

            if subset != next_subset and subset in subset_results:
                subset_acc = subset_stats[subset]['correct'] / subset_stats[subset]['total']
                logger.info(f"=== Subset {subset} completed: {subset_stats[subset]['correct']}/{subset_stats[subset]['total']} = {subset_acc:.3f} ===")

                subset_file = f'RIM/rewardbench_subset_{subset.replace("/", "_")}_{timestamp}.json'
                with open(subset_file, 'w') as f:
                    json.dump({
                        'subset': subset,
                        'accuracy': subset_acc,
                        'correct': subset_stats[subset]['correct'],
                        'total': subset_stats[subset]['total'],
                        'details': subset_results[subset]
                    }, f, indent=2)
                logger.info(f"Saved subset results to {subset_file}")

        except Exception as e:
            logger.error(f"Error processing sample {i} (ID: {sample['id']}): {e}", exc_info=True)

    final_accuracy = correct / total if total > 0 else 0
    logger.info(f"\n=== Final Results ===")
    logger.info(f"Total samples: {total}")
    logger.info(f"Correct: {correct}")
    logger.info(f"Accuracy: {final_accuracy:.3f}")

    logger.info(f"\n=== Results by Subset ===")
    subset_accuracies = {}
    for subset, stats in sorted(subset_stats.items()):
        subset_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        subset_accuracies[subset] = subset_acc
        logger.info(f"{subset}: {stats['correct']}/{stats['total']} = {subset_acc:.3f}")

    final_results_file = f'RIM/rewardbench_results_{timestamp}.json'
    with open(final_results_file, 'w') as f:
        json.dump({
            'accuracy': final_accuracy,
            'total_samples': total,
            'correct': correct,
            'subset_accuracies': subset_accuracies,
            'subset_stats': subset_stats,
            'details': results
        }, f, indent=2)

    logger.info(f"Final results saved to {final_results_file}")
    logger.info(f"Log file saved to {log_file}")

    return final_accuracy


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    if not os.getenv('GEMINI_API_KEY'):
        print("Error: GEMINI_API_KEY not found in .env file")
        exit(1)

    test_rim_on_rewardbench(num_samples=1870)
