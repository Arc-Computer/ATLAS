from __future__ import annotations

import shlex
import subprocess
from typing import Any, Dict

from .base import PromptInput, PromptOutput, expand_env


class CliAgentClient:
    def __init__(self, params: Dict[str, Any]):
        config = expand_env(params)
        self.command = config["command"]
        self.timeout = config.get("timeout", 300)

    def __call__(self, prompts: PromptInput) -> PromptOutput:
        if isinstance(prompts, str):
            return self._run_single(prompts)
        return [self._run_single(prompt) for prompt in prompts]

    def _run_single(self, prompt: str) -> str:
        escaped_prompt = shlex.quote(prompt)
        cmd = self.command.replace("{prompt}", escaped_prompt)
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "Command timed out"
        except Exception as exc:
            return f"Command error: {exc}"
        if result.returncode != 0:
            return f"Command failed: {result.stderr}"
        return result.stdout.strip()


def load(params: Dict[str, Any]) -> CliAgentClient:
    return CliAgentClient(params)
