from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
from typing import Any, Dict, Union

from .base import PromptInput, PromptOutput, expand_env


class PythonCallableClient:
    def __init__(self, params: Dict[str, Any]):
        config = expand_env(params)
        self.module_path = config["module_path"]
        self.function_name = config["function_name"]
        self.class_name = config.get("class_name")
        self.module_base_path = config.get("module_base_path")
        self.process_input = config.get("process_input", "raw")
        self.input_key = config.get("input_key", "input")
        self.process_response = config.get("process_response", "raw")
        self.suppress_output = config.get("suppress_output", False)
        self.capture_path = config.get("capture_output_to_file")
        self.agent_function = self._load_callable()

    def __call__(self, prompts: PromptInput) -> PromptOutput:
        if isinstance(prompts, str):
            return self._run_single(prompts)
        return [self._run_single(prompt) for prompt in prompts]

    def _load_callable(self):
        if self.module_base_path:
            base_path = os.path.abspath(self.module_base_path)
            if base_path not in sys.path:
                sys.path.insert(0, base_path)
        module_name = os.path.splitext(os.path.basename(self.module_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, os.path.abspath(self.module_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {self.module_path}")
        module = importlib.util.module_from_spec(spec)
        module.__file__ = os.path.abspath(self.module_path)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        if self.class_name:
            agent_class = getattr(module, self.class_name)
            agent_instance = agent_class()
            crew_instance = agent_instance.crew()
            return lambda inputs: getattr(crew_instance, self.function_name)(inputs=inputs)
        return getattr(module, self.function_name)

    def _run_single(self, prompt: str) -> str:
        agent_input = self._prepare_input(prompt)
        result = self._invoke(agent_input)
        return self._format_output(result)

    def _prepare_input(self, prompt: str) -> Union[str, Dict[str, Any]]:
        if self.process_input == "dict_wrapper":
            try:
                parsed = json.loads(prompt)
            except json.JSONDecodeError:
                parsed = prompt
            return {self.input_key: parsed}
        return prompt

    def _invoke(self, agent_input: Any) -> Any:
        if not self.suppress_output:
            return self.agent_function(agent_input)
        captured_out = io.StringIO()
        captured_err = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = captured_out
            sys.stderr = captured_err
            result = self.agent_function(agent_input)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        if self.capture_path:
            os.makedirs(os.path.dirname(self.capture_path), exist_ok=True)
            with open(self.capture_path, "a") as handle:
                handle.write("=== Agent Output ===\n")
                handle.write(captured_out.getvalue())
                errors = captured_err.getvalue()
                if errors:
                    handle.write("\n=== Errors ===\n")
                    handle.write(errors)
                handle.write("\n=== End ===\n\n")
        return result

    def _format_output(self, result: Any) -> str:
        if self.process_response == "json":
            return self._json_response(result)
        if result is None:
            return ""
        return str(result)

    def _json_response(self, result: Any) -> str:
        if result is None:
            return "{}"
        if hasattr(result, "json_dict") and result.json_dict:
            return json.dumps(result.json_dict)
        if hasattr(result, "raw") and result.raw:
            raw_content = result.raw
            if isinstance(raw_content, str):
                try:
                    json.loads(raw_content)
                    return raw_content
                except json.JSONDecodeError:
                    match = self._extract_json_block(raw_content)
                    return match or (raw_content if raw_content else "{}")
            return str(raw_content) if raw_content else "{}"
        if hasattr(result, "result") and result.result is not None:
            return result.result
        if hasattr(result, "__dict__"):
            return json.dumps(result.__dict__)
        try:
            return json.dumps(result)
        except TypeError:
            return str(result)

    def _extract_json_block(self, text: str) -> str:
        import re

        pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.findall(pattern, text, re.DOTALL)
        for candidate in reversed(matches):
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue
        return "{}"


def load(params: Dict[str, Any]) -> PythonCallableClient:
    return PythonCallableClient(params)
