import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Union

from openai import OpenAI


class OpenAIAssistantWrapper:
    def __init__(self, config: Dict[str, Any]):
        self.config = self._expand_env_vars(config)
        self.client = OpenAI(api_key=self.config["api_key"])
        self.assistant_id = self.config.get("assistant_id")
        self.timeout = self.config.get("timeout", 300)
        self.max_workers = self.config.get("max_workers", 10)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        if not self.assistant_id:
            self.assistant_id = self._create_assistant()

    def _expand_env_vars(self, value: Any) -> Any:
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.environ.get(env_var, value)
        elif isinstance(value, dict):
            return {k: self._expand_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._expand_env_vars(item) for item in value]
        return value

    def _create_assistant(self) -> str:
        response_format = self.config.get("response_format", {"type": "text"})
        assistant = self.client.beta.assistants.create(
            name=self.config.get("name", "ATLAS Agent"),
            instructions=self.config.get("instructions", ""),
            model=self.config.get("model", "gpt-4o-mini"),
            response_format=response_format,
        )
        return assistant.id

    def _process_single(self, prompt: str) -> str:
        thread = self.client.beta.threads.create()

        self.client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content=prompt
        )

        run = self.client.beta.threads.runs.create(
            thread_id=thread.id, assistant_id=self.assistant_id
        )

        start_time = time.time()
        while run.status not in ["completed", "failed", "cancelled"]:
            if time.time() - start_time > self.timeout:
                raise TimeoutError(f"Run timed out after {self.timeout}s")
            time.sleep(1)
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread.id, run_id=run.id
            )

        if run.status != "completed":
            raise RuntimeError(f"Run failed: {run.status}")

        messages = self.client.beta.threads.messages.list(
            thread_id=thread.id,
            order="desc"
        )

        for message in messages.data:
            if message.role == "assistant":
                content = message.content[0]

                if hasattr(content, 'text'):
                    response = content.text.value
                elif hasattr(content, 'output_json'):
                    response = json.dumps(content.output_json)
                else:
                    response = str(content)

                return self._extract_output(response)

        raise RuntimeError("No assistant message found in thread")

    def _extract_output(self, response: str) -> str:
        extract_config = self.config.get("output_extraction", {})
        extract_type = extract_config.get("type", "direct")

        if extract_type == "json_field":
            try:
                data = json.loads(response)
                field_path = extract_config.get("field_path", "solution")
                for key in field_path.split("."):
                    if isinstance(data, dict):
                        data = data.get(key, "")
                template = extract_config.get("format_template", "{value}")
                return template.format(value=str(data))
            except json.JSONDecodeError:
                return response
        else:
            return response

    def __call__(self, prompts: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(prompts, str):
            prompts = [prompts]
            single = True
        else:
            single = False

        futures = [self.executor.submit(self._process_single, p) for p in prompts]
        responses = []

        for future in futures:
            try:
                responses.append(future.result(timeout=self.timeout))
            except Exception as e:
                responses.append(f"Error: {str(e)}")

        return responses[0] if single else responses