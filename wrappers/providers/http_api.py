from __future__ import annotations

import copy
from typing import Any, Dict

import requests

from .base import PromptInput, PromptOutput, expand_env


class HttpAgentClient:
    def __init__(self, params: Dict[str, Any]):
        config = expand_env(params)
        self.endpoint = config["endpoint"]
        self.method = config.get("method", "POST").upper()
        self.headers = config.get("headers", {})
        self.timeout = config.get("timeout", 300)
        self.prompt_field = config.get("prompt_field", "prompt")
        self.response_field = config.get("response_field", "response")
        self.request_template = config.get("request_template", {})
        self.query_template = config.get("query_template", {})

    def __call__(self, prompts: PromptInput) -> PromptOutput:
        if isinstance(prompts, str):
            return self._run_single(prompts)
        return [self._run_single(prompt) for prompt in prompts]

    def _run_single(self, prompt: str) -> str:
        payload = copy.deepcopy(self.request_template)
        if self.prompt_field:
            payload[self.prompt_field] = prompt
        params = copy.deepcopy(self.query_template)
        try:
            response = requests.request(
                self.method,
                self.endpoint,
                headers=self.headers,
                json=payload if payload else None,
                params=params if params else None,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            return f"Error calling API: {exc}"
        try:
            data = response.json()
        except ValueError:
            return response.text.strip()
        return self._extract_field(data, self.response_field)

    def _extract_field(self, data: Any, field_path: str) -> str:
        if not field_path:
            return str(data) if data is not None else ""
        current = data
        for key in field_path.split("."):
            if isinstance(current, dict):
                current = current.get(key)
            else:
                return str(current) if current is not None else ""
        return str(current) if current is not None else ""


def load(params: Dict[str, Any]) -> HttpAgentClient:
    return HttpAgentClient(params)
