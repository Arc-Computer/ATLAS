from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

try:
    from agents import Agent, Runner, function_tool
except ImportError as exc:
    raise ImportError(
        "OpenAI Agents SDK not found. Install it with: pip install openai-agents"
    ) from exc

from .base import PromptInput, PromptOutput, expand_env


class OpenAISDKClient:
    def __init__(self, params: Dict[str, Any]):
        config = expand_env(params)
        self.agent = Agent(
            name=config.get("name", "Agent"),
            instructions=config.get("instructions", ""),
            tools=self._load_tools(config.get("tools", [])),
            handoffs=self._load_handoffs(config.get("handoffs", [])),
            model=config.get("model"),
        )

    def __call__(self, prompts: PromptInput) -> PromptOutput:
        if isinstance(prompts, str):
            return self._run_single(prompts)
        with ThreadPoolExecutor(max_workers=10) as executor:
            return list(executor.map(self._run_single, prompts))

    def _run_single(self, prompt: str) -> str:
        result = Runner.run_sync(self.agent, prompt)
        output = result.final_output
        if isinstance(output, dict):
            return json.dumps(output)
        if isinstance(output, str):
            return output
        return str(output)

    def _load_tools(self, configs: list[Dict[str, Any]]):
        tools = []
        for item in configs:
            if item.get("type") == "function":
                module_path = item.get("module_path")
                function_name = item.get("function_name")
                if module_path and function_name:
                    import importlib.util

                    spec = importlib.util.spec_from_file_location("custom_tool", module_path)
                    if spec is None or spec.loader is None:
                        raise ImportError(f"Cannot load module from {module_path}")
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    func = getattr(module, function_name)
                    tools.append(function_tool(func))
        return tools

    def _load_handoffs(self, configs: list[Dict[str, Any]]):
        handoffs = []
        for item in configs:
            handoffs.append(
                Agent(
                    name=item.get("name", "Handoff"),
                    instructions=item.get("instructions", ""),
                )
            )
        return handoffs


def load(params: Dict[str, Any]) -> OpenAISDKClient:
    return OpenAISDKClient(params)
