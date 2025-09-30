from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

from openai import OpenAI

from .base import PromptInput, PromptOutput, expand_env


class OpenAIAssistantClient:
    def __init__(self, params: Dict[str, Any]):
        config = expand_env(params)
        self.client = OpenAI(api_key=config["api_key"])
        self.timeout = config.get("timeout", 300)
        self.max_workers = config.get("max_workers", 10)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.response_format = config.get("response_format", {"type": "text"})
        self.extract_config = config.get("output_extraction", {})
        self.assistant_id = config.get("assistant_id") or self._create_assistant(config)

    def __call__(self, prompts: PromptInput) -> PromptOutput:
        if isinstance(prompts, str):
            return self._run_single(prompts)
        futures = [self.executor.submit(self._run_single, prompt) for prompt in prompts]
        results: list[str] = []
        for future in futures:
            try:
                results.append(future.result(timeout=self.timeout))
            except Exception as exc:
                results.append(f"Error: {exc}")
        return results

    def _create_assistant(self, config: Dict[str, Any]) -> str:
        assistant = self.client.beta.assistants.create(
            name=config.get("name", "ATLAS Agent"),
            instructions=config.get("instructions", ""),
            model=config.get("model", "gpt-4o-mini"),
            response_format=self.response_format,
        )
        return assistant.id

    def _run_single(self, prompt: str) -> str:
        thread = self.client.beta.threads.create()
        self.client.beta.threads.messages.create(thread_id=thread.id, role="user", content=prompt)
        run = self.client.beta.threads.runs.create(thread_id=thread.id, assistant_id=self.assistant_id)
        start = time.time()
        while run.status not in {"completed", "failed", "cancelled"}:
            if time.time() - start > self.timeout:
                raise TimeoutError(f"Run timed out after {self.timeout}s")
            time.sleep(1)
            run = self.client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run.status != "completed":
            raise RuntimeError(f"Run failed: {run.status}")
        messages = self.client.beta.threads.messages.list(thread_id=thread.id, order="desc")
        for message in messages.data:
            if message.role == "assistant":
                content = message.content[0]
                if hasattr(content, "text"):
                    return self._extract_output(content.text.value)
                if hasattr(content, "output_json"):
                    payload = json.dumps(content.output_json)
                    return self._extract_output(payload)
                return self._extract_output(str(content))
        raise RuntimeError("No assistant message found in thread")

    def _extract_output(self, value: str) -> str:
        extract_type = self.extract_config.get("type", "direct")
        if extract_type == "json_field":
            try:
                data = json.loads(value)
                field_path = self.extract_config.get("field_path", "")
                for key in field_path.split(".") if field_path else []:
                    if isinstance(data, dict):
                        data = data.get(key, "")
                template = self.extract_config.get("format_template", "{value}")
                return template.format(value=str(data))
            except json.JSONDecodeError:
                return value
        return value


def load(params: Dict[str, Any]) -> OpenAIAssistantClient:
    return OpenAIAssistantClient(params)
