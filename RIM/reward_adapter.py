from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

import yaml
from pathlib import Path
from RIM.rim import RewardInterpretationModel
from RIM.judges import AccuracyJudge, HelpfulnessJudge, ProcessJudge, DiagnosticJudge
from atlas_core.runtime import (
    AtlasRewardBreakdown,
    AtlasJudgeBreakdown,
    AtlasJudgeSample,
)


@dataclass
class RIMEvaluation:
    """Structured result for single-sample reward evaluations."""

    score: float
    rationale: str
    judge_scores: Dict[str, float]
    judge_explanations: Dict[str, str]
    reward: AtlasRewardBreakdown
    extra: Optional[Dict[str, Any]] = None


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
        self._last_structured_rewards: List[AtlasRewardBreakdown] = []

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
        structured_rewards: List[AtlasRewardBreakdown] = []

        active_judges = [judge for judge, active in self.rim.active_judges.items() if active]
        if not active_judges:
            active_judges = list(all_rewards.keys())

        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            trajectory = self._build_trajectory(
                prompt=prompt,
                completion=completion,
                index=i,
                **kwargs
            )

            result = self.rim.evaluate(trajectory)
            self._apply_consistency_rules(result['rewards'], result['explanations'])
            structured_rewards.append(self._build_reward_breakdown(result, active_judges))

            for judge in result['rewards']:
                if judge in all_rewards:
                    all_rewards[judge].append(result['rewards'][judge])
                    all_explanations[judge].append(result['explanations'][judge])

        self._last_structured_rewards = structured_rewards

        combined_rewards = []
        for i in range(len(prompts)):
            sample_rewards = [all_rewards[judge][i] for judge in active_judges if judge in all_rewards]
            combined_rewards.append(sum(sample_rewards) / len(sample_rewards) if sample_rewards else 0.0)

        if return_info_dict:
            info_dict = []
            for i in range(len(prompts)):
                sample_info = {
                    'rewards': {judge: all_rewards[judge][i] for judge in active_judges if judge in all_rewards},
                    'explanations': {judge: all_explanations[judge][i] for judge in active_judges if judge in all_explanations},
                    'structured_reward': structured_rewards[i],
                }
                info_dict.append(sample_info)

            if return_raw_tensors:
                return combined_rewards, info_dict, [{}] * len(prompts)
            return combined_rewards, info_dict

        return combined_rewards


    def _apply_consistency_rules(self, rewards: Dict[str, float], explanations: Dict[str, str]) -> None:
        if not rewards:
            return

        def _clamp(value: float) -> float:
            return max(0.0, min(1.0, value))

        def _update(judge: str, value: float, note: str) -> None:
            rewards[judge] = _clamp(value)
            if not note:
                return
            explanation = explanations.get(judge, '')
            note_text = f"[Consistency] {note}"
            explanations[judge] = f"{explanation}\n{note_text}".strip() if explanation else note_text

        accuracy = rewards.get('accuracy')
        process = rewards.get('process')
        helpfulness = rewards.get('helpfulness')
        diagnostic = rewards.get('diagnostic')

        if accuracy is not None and process is not None and accuracy < process - 0.1:
            _update('process', min(process, accuracy + 0.1), 'Process capped by accuracy + 0.1')
            process = rewards.get('process')

        if process is not None and helpfulness is not None and process < 0.5 and helpfulness > 0.5:
            _update('helpfulness', min(helpfulness, 0.5), 'Helpfulness limited when process < 0.5')
            helpfulness = rewards.get('helpfulness')

        if diagnostic is not None and helpfulness is not None and diagnostic < helpfulness - 0.1:
            _update('helpfulness', min(helpfulness, diagnostic + 0.1), 'Helpfulness capped by diagnostic + 0.1')

        for judge in ('accuracy', 'process', 'helpfulness', 'diagnostic'):
            if judge in rewards:
                rewards[judge] = _clamp(rewards[judge])

    def evaluate(
        self,
        prompt: str,
        response: str,
        **kwargs
    ) -> RIMEvaluation:
        """Evaluate a single prompt/response pair and return structured results.

        This is a thin wrapper around the batched ``__call__`` interface used by
        training pipelines. It exists for developer ergonomics in quickstarts or
        ad-hoc evaluations where batching is unnecessary.
        """

        batched_keys = {
            'ground_truths',
            'baseline_solutions',
            'student_plans',
            'teacher_traces',
            'student_traces',
        }

        call_kwargs = {}
        for key, value in kwargs.items():
            if key in batched_keys and not isinstance(value, list):
                call_kwargs[key] = [value]
            else:
                call_kwargs[key] = value

        return_raw_tensors = call_kwargs.pop('return_raw_tensors', False)

        raw_result = self(
            prompts=[prompt],
            completions=[response],
            return_info_dict=True,
            return_raw_tensors=return_raw_tensors,
            **call_kwargs,
        )

        if return_raw_tensors:
            rewards, info_dict, raw_tensors = cast(
                Tuple[List[float], List[Dict[str, Any]], Any],
                raw_result,
            )
        else:
            rewards, info_dict = cast(
                Tuple[List[float], List[Dict[str, Any]]],
                raw_result,
            )
            raw_tensors = None

        score = rewards[0] if rewards else 0.0
        sample_info = info_dict[0] if info_dict else {'rewards': {}, 'explanations': {}, 'structured_reward': None}

        judge_scores = sample_info.get('rewards', {}) or {}
        judge_explanations = sample_info.get('explanations', {}) or {}
        structured_reward = sample_info.get('structured_reward')
        if structured_reward is None:
            structured_reward = AtlasRewardBreakdown(score=score, judges=[], rationale=None, raw=None)

        active_judges = [
            judge for judge, active in self.rim.active_judges.items()
            if active and judge in judge_explanations
        ]
        if not active_judges:
            active_judges = list(judge_explanations.keys())

        rationale_parts = [
            f"{judge}: {judge_explanations[judge]}".strip()
            for judge in active_judges
            if judge in judge_explanations and judge_explanations[judge]
        ]
        rationale = "\n".join(part for part in rationale_parts if part).strip()

        return RIMEvaluation(
            score=score,
            rationale=rationale,
            judge_scores=judge_scores,
            judge_explanations=judge_explanations,
            reward=structured_reward,
            extra={'raw_tensors': raw_tensors, 'info': sample_info}
        )

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

    def _build_reward_breakdown(
        self,
        result: Dict[str, Any],
        active_judges: List[str],
    ) -> AtlasRewardBreakdown:
        records = result.get('records') or {}
        judges: List[AtlasJudgeBreakdown] = []
        for judge in active_judges:
            record = records.get(judge, {})
            score = record.get('score', result['rewards'].get(judge, 0.0))
            rationale = record.get('rationale', result['explanations'].get(judge, ''))
            principles = record.get('principles', [])
            samples_payload = record.get('samples', [])
            samples = [
                AtlasJudgeSample(
                    score=sample.get('score', 0.0),
                    rationale=sample.get('rationale', ''),
                    principles=sample.get('principles', []),
                    uncertainty=sample.get('uncertainty'),
                    temperature=sample.get('temperature'),
                )
                for sample in samples_payload
            ]
            judges.append(
                AtlasJudgeBreakdown(
                    identifier=judge,
                    score=score,
                    rationale=rationale,
                    principles=principles,
                    samples=samples,
                    escalated=record.get('escalated', False),
                    escalation_reason=record.get('escalation_reason'),
                )
            )
        aggregated_score = (
            sum(j.score for j in judges) / len(judges) if judges else 0.0
        )
        rationale_text = "\n".join(j.rationale for j in judges if j.rationale) or None
        return AtlasRewardBreakdown(
            score=aggregated_score,
            judges=judges,
            rationale=rationale_text,
            raw=result,
        )

    @property
    def last_structured_rewards(self) -> List[AtlasRewardBreakdown]:
        """Return the structured rewards from the most recent batched evaluation."""

        return getattr(self, "_last_structured_rewards", [])
