from typing import List, Dict, Any, Tuple, Optional
import yaml
from pathlib import Path
from RIM.rim import RewardInterpretationModel
from RIM.judges import AccuracyJudge, HelpfulnessJudge, ProcessJudge, DiagnosticJudge


class RIMReward:
    def __init__(
        self,
        student_model=None,
        teacher_model=None,
        tokenizer=None,
        config_path: str = 'configs/rim_config.yaml'
    ):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.__name__ = 'RIMReward'

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.rim = RewardInterpretationModel(config['rim'])
        self.judges = {
            'accuracy': AccuracyJudge(),
            'helpfulness': HelpfulnessJudge(),
            'process': ProcessJudge(),
            'diagnostic': DiagnosticJudge()
        }

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        return_info_dict: bool = False,
        return_raw_tensors: bool = False,
        **kwargs
    ):
        all_rewards = {
            'accuracy': [],
            'helpfulness': [],
            'process': [],
            'diagnostic': []
        }
        all_explanations = {
            'accuracy': [],
            'helpfulness': [],
            'process': [],
            'diagnostic': []
        }

        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            trajectory = self._build_trajectory(
                prompt=prompt,
                completion=completion,
                index=i,
                **kwargs
            )

            result = self.rim.evaluate(trajectory)

            for judge in result['rewards']:
                if judge in all_rewards:
                    all_rewards[judge].append(result['rewards'][judge])
                    all_explanations[judge].append(result['explanations'][judge])

        active_judges = [judge for judge, active in self.rim.active_judges.items() if active]

        combined_rewards = []
        for i in range(len(prompts)):
            sample_rewards = [all_rewards[judge][i] for judge in active_judges if judge in all_rewards]
            combined_rewards.append(sum(sample_rewards) / len(sample_rewards) if sample_rewards else 0.0)

        if return_info_dict:
            info_dict = []
            for i in range(len(prompts)):
                sample_info = {
                    'rewards': {judge: all_rewards[judge][i] for judge in active_judges if judge in all_rewards},
                    'explanations': {judge: all_explanations[judge][i] for judge in active_judges if judge in all_explanations}
                }
                info_dict.append(sample_info)

            if return_raw_tensors:
                return combined_rewards, info_dict, [{}] * len(prompts)
            return combined_rewards, info_dict

        return combined_rewards

    def _get_reward_descriptions(self) -> Dict[str, str]:
        return {
            'accuracy': 'Measures alignment between student answer and ground truth',
            'helpfulness': 'Measures whether teaching was helpful to the student',
            'process': 'Measures completeness and quality of student planning',
            'diagnostic': 'Measures teacher understanding of student mistakes'
        }

    def _get_consistency_rules(self) -> List[str]:
        return [
            'If accuracy < 0.5, process cannot exceed accuracy + 0.1',
            'If process < 0.5 and gaps not addressed, helpfulness capped at 0.5',
            'If diagnostic < 0.5, helpfulness cannot exceed diagnostic + 0.1',
            'Contradictions reduce accuracy by 0.3',
            'Safety violations reduce all affected scores by 0.3'
        ]

    def _build_trajectory(
        self,
        prompt: str,
        completion: str,
        index: int,
        **kwargs
    ) -> Dict[str, Any]:
        trajectory = {
            'question': prompt,
            'student_answer': completion,
            'ground_truth': kwargs.get('ground_truths', [''])[index] if 'ground_truths' in kwargs else '',
            'student_baseline_answer': kwargs.get('baseline_solutions', [''])[index] if 'baseline_solutions' in kwargs else '',
            'student_plan': kwargs.get('student_plans', [''])[index] if 'student_plans' in kwargs else '',
            'teacher_trace': kwargs.get('teacher_traces', [''])[index] if 'teacher_traces' in kwargs else '',
            'student_trace': kwargs.get('student_traces', [''])[index] if 'student_traces' in kwargs else '',
            'constitution_principles': kwargs.get('constitution_principles', '')
        }

        for key in kwargs:
            if key not in trajectory and not key.startswith('cached_'):
                if isinstance(kwargs[key], list) and len(kwargs[key]) > index:
                    trajectory[key] = kwargs[key][index]

        return trajectory

