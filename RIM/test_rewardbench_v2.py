import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datasets import load_dataset
import numpy as np
from typing import Dict
from RIM.rim import RewardInterpretationModel
import json
import yaml
import os
import logging
from datetime import datetime


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

    rim = RewardInterpretationModel(config['rim'])

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
            config['rim']['active_judges'] = {'accuracy': True, 'helpfulness': False, 'process': False, 'diagnostic': False}
            rim.active_judges = config['rim']['active_judges']

            result = rim._evaluate_single_reward(trajectory, 'accuracy')

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