import json
from pathlib import Path

import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from RIM.reward_adapter import RIMReward
from atlas_core.runtime import AtlasRewardBreakdown


@pytest.fixture()
def stub_config(tmp_path: Path) -> Path:
    config = {
        "rim": {
            "temperatures": [0.2, 0.5],
            "variance_threshold": 1.0,
            "active_judges": {"accuracy": True},
            "models": {
                "small_model": "stub-small",
                "large_model": "stub-large",
            },
            "parallel_execution": {"max_workers": 1},
        }
    }
    path = tmp_path / "rim_config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def stubbed_model_interface(monkeypatch):
    def _call_model(model_name, prompt, temperature, max_tokens):
        if model_name == "stub-small":
            return json.dumps(
                {
                    "score": 0.8,
                    "uncertainty": 0.1,
                    "rationale": "Small model rationale",
                    "principles": [
                        {"name": "Execution", "weight": 1.0, "description": "step quality"}
                    ],
                }
            )
        return json.dumps(
            {
                "score": 0.75,
                "uncertainty": 0.2,
                "rationale": "Arbiter fallback",
                "principles": [
                    {"name": "Consistency", "weight": 1.0, "description": "arbiter"}
                ],
            }
        )

    monkeypatch.setattr(
        "RIM.model_interface.model_interface.call_model",
        _call_model,
    )
    monkeypatch.setattr(
        "RIM.judge_specs.get_judge_prompt",
        lambda reward_type: (
            "Question: {question}\n"
            "Answer: {student_answer}\n"
            "Principles: {constitution_principles}"
        ),
    )
    monkeypatch.setattr(
        "RIM.rim.RewardInterpretationModel._build_judge_prompt",
        lambda self, trajectory, reward_type: "prompt",
    )


def test_rim_reward_emits_structured_breakdown(stub_config: Path):
    reward_fn = RIMReward(config_path=str(stub_config))

    evaluation = reward_fn.evaluate(
        prompt="What is 2+2?",
        response="The answer is <solution>4</solution>.",
    )

    assert isinstance(evaluation.reward, AtlasRewardBreakdown)
    assert evaluation.reward.score == pytest.approx(0.8)
    assert evaluation.reward.judges, "expected judge breakdown"
    judge = evaluation.reward.judges[0]
    assert judge.identifier == "accuracy"
    assert judge.score == pytest.approx(0.8)
    assert judge.rationale
    assert judge.samples, "expected per-sample metadata"
    sample = judge.samples[0]
    assert sample.score == pytest.approx(0.8)
    assert sample.uncertainty == pytest.approx(0.1)

    last_structured = reward_fn.last_structured_rewards
    assert last_structured, "last_structured_rewards should reflect latest batch"
    assert last_structured[0].score == pytest.approx(0.8)
