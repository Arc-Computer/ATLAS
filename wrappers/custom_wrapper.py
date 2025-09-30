from typing import Any, Dict, List, Union
import copy
import importlib.util
import shlex
import subprocess
import requests


class CustomWrapper:
    """Wrapper for any existing agent - API, CLI, Python function, etc."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.integration_type = config["integration_type"]
        self.suppress_output = config.get("suppress_output", False)

        if self.integration_type == "python_function":
            self._setup_python_function()
        elif self.integration_type == "http_api":
            self.endpoint = config["endpoint"]
            self.headers = config.get("headers", {})
            self.request_template = config.get("request_template", {})
            self.prompt_field = config.get("prompt_field", "prompt")
            self.response_field = config.get("response_field", "response")
        elif self.integration_type == "cli_command":
            self.command = config["command"]

    def _setup_python_function(self):
        import sys
        import os
        module_path = self.config["module_path"]
        function_name = self.config["function_name"]
        class_name = self.config.get("class_name")
        module_base_path = self.config.get("module_base_path")

        if module_base_path:
            abs_base_path = os.path.abspath(module_base_path)
            if abs_base_path not in sys.path:
                sys.path.insert(0, abs_base_path)

        import os
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        spec = importlib.util.spec_from_file_location(
            module_name,
            os.path.abspath(module_path),
            submodule_search_locations=[]
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        module.__file__ = os.path.abspath(module_path)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if class_name:
            agent_class = getattr(module, class_name)
            agent_instance = agent_class()
            crew_instance = agent_instance.crew()
            self.agent_function = lambda inputs: getattr(crew_instance, function_name)(inputs=inputs)
        else:
            self.agent_function = getattr(module, function_name)

    def __call__(self, prompts: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(prompts, str):
            prompts = [prompts]
            single = True
        else:
            single = False

        responses = []
        for prompt in prompts:
            response = self._call_agent(prompt)
            responses.append(response)

        return responses[0] if single else responses

    def _suppress_output(self, func, *args, **kwargs):
        """Universal output suppression for any function call"""
        if not self.suppress_output:
            return func(*args, **kwargs)

        import sys
        import io
        import os
        import logging

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_log_level = logging.root.level

        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        logging.root.setLevel(logging.CRITICAL)

        captured_stdout = sys.stdout
        captured_stderr = sys.stderr

        try:
            result = func(*args, **kwargs)

            stdout_content = captured_stdout.getvalue()
            stderr_content = captured_stderr.getvalue()

            if self.config.get("capture_output_to_file"):
                # Ensure directory exists
                import os
                log_file = self.config["capture_output_to_file"]
                os.makedirs(os.path.dirname(log_file), exist_ok=True)

                with open(log_file, "a") as f:
                    f.write(f"=== Agent Output ===\n")
                    f.write(stdout_content)
                    if stderr_content:
                        f.write(f"\n=== Errors ===\n")
                        f.write(stderr_content)
                    f.write(f"\n=== End ===\n\n")

            return result
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            logging.root.setLevel(old_log_level)

    def _call_agent(self, prompt: str) -> str:
        if self.integration_type == "python_function":
            process_input = self.config.get("process_input", "raw")
            if process_input == "dict_wrapper":
                input_key = self.config.get("input_key", "input")
                import json
                try:
                    parsed = json.loads(prompt) if isinstance(prompt, str) else prompt
                    agent_input = {input_key: parsed}
                except (json.JSONDecodeError, TypeError):
                    agent_input = {input_key: prompt}
            else:
                agent_input = prompt

            result = self._suppress_output(self.agent_function, agent_input)

            process_response = self.config.get("process_response", "raw")
            if process_response == "raw":
                return str(result) if result is not None else ""
            elif process_response == "json":
                import json

                if hasattr(result, 'json_dict') and result.json_dict:
                    return json.dumps(result.json_dict)
                elif hasattr(result, 'raw') and result.raw:
                    raw_content = result.raw
                    if isinstance(raw_content, str):
                        # First check if the entire raw content is valid JSON
                        try:
                            json.loads(raw_content)
                            # If it's valid JSON, return it as-is
                            return raw_content
                        except json.JSONDecodeError:
                            # If not, try to extract JSON from it
                            try:
                                json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                                import re
                                matches = re.findall(json_pattern, raw_content, re.DOTALL)
                                if matches:
                                    for match in reversed(matches):
                                        try:
                                            json.loads(match)
                                            return match
                                        except json.JSONDecodeError:
                                            continue
                            except:
                                pass
                            return raw_content if raw_content else "{}"
                    return str(raw_content) if raw_content else "{}"
                elif hasattr(result, 'result'):
                    return result.result if result.result is not None else "{}"
                elif hasattr(result, '__dict__'):
                    return json.dumps(result.__dict__) if result is not None else "{}"
                else:
                    return json.dumps(result) if result is not None else "{}"
            else:
                return str(result) if result is not None else ""

        elif self.integration_type == "http_api":
            import copy
            payload = copy.deepcopy(self.request_template)
            payload[self.prompt_field] = prompt

            try:
                response = requests.post(
                    self.endpoint,
                    json=payload,
                    headers=self.headers,
                    timeout=self.config.get("timeout", 300)
                )
                response.raise_for_status()

                try:
                    result = response.json()
                except ValueError:
                    return response.text.strip()

                return self._extract_field(result, self.response_field)
            except requests.exceptions.RequestException as e:
                return f"Error calling API: {str(e)}"
            except (KeyError, TypeError) as e:
                return f"Error parsing response: {str(e)}"

        elif self.integration_type == "cli_command":
            try:
                escaped_prompt = shlex.quote(prompt)
                cmd = self.command.replace("{prompt}", escaped_prompt)

                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.config.get("timeout", 300),
                    check=False
                )

                if result.returncode != 0:
                    return f"Command failed: {result.stderr}"
                return result.stdout.strip()
            except subprocess.TimeoutExpired:
                return "Command timed out"
            except Exception as e:
                return f"Command error: {str(e)}"

    def _extract_field(self, data: dict, field_path: str) -> str:
        try:
            for key in field_path.split("."):
                if isinstance(data, dict):
                    data = data.get(key)
                    if data is None:
                        return ""
                else:
                    return str(data)
            return str(data) if data is not None else ""
        except (AttributeError, TypeError):
            return ""