"""Demonstrate retry awareness and structured errors from the Atlas SDK."""

from __future__ import annotations

import json
from typing import Any

from atlas import run
from atlas.agent.registry import AdapterError
from atlas.types import StepResult

CONFIG_PATH = "configs/examples/sdk_quickstart.yaml"


def pretty_attempt(step_result: StepResult) -> dict[str, Any]:
    return {
        "step_id": step_result.step_id,
        "output": step_result.output[:200],
        "score": step_result.evaluation.get("reward", {}).get("score"),
        "attempts": step_result.attempts,
    }


def main() -> None:
    try:
        result = run(
            task="List two recent breakthroughs in AI safety research",
            config_path=CONFIG_PATH,
        )
    except AdapterError as exc:
        print("Adapter failed:", exc)
        return
    except Exception as exc:  # Catch anything unexpected so tests can continue
        print("Unexpected failure:", exc)
        return

    print("Final answer:\n", result.final_answer)
    print("\nStep summary:")
    for step in result.step_results:
        print(json.dumps(pretty_attempt(step), indent=2))


if __name__ == "__main__":
    main()
