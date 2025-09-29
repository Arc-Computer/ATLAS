import json
import os
from typing import Any, Dict, List, Union

try:
    from agents import Agent, Runner
except ImportError:
    raise ImportError(
        "OpenAI Agents SDK not found. Install it with: "
        "pip install openai-agents"
    )


class OpenAISDKWrapper:
    def __init__(self, config: Dict[str, Any]):
        self.config = self._expand_env_vars(config)
        self.agent = self._create_agent()

    def _expand_env_vars(self, value: Any) -> Any:
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.environ.get(env_var, value)
        elif isinstance(value, dict):
            return {k: self._expand_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._expand_env_vars(item) for item in value]
        return value

    def _create_agent(self) -> Agent:
        return Agent(
            name=self.config.get("name", "Agent"),
            instructions=self.config.get("instructions", ""),
            tools=self._load_tools(),
            handoffs=self._load_handoffs(),
            model=self.config.get("model"),
        )

    def _load_tools(self):
        tools = []
        for tool_config in self.config.get("tools", []):
            if tool_config["type"] == "function":
                from agents import function_tool
                import importlib.util

                if "module_path" in tool_config:
                    spec = importlib.util.spec_from_file_location(
                        "custom_tool",
                        tool_config["module_path"]
                    )
                    if spec is None or spec.loader is None:
                        raise ImportError(f"Cannot load module from {tool_config['module_path']}")
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    func = getattr(module, tool_config["function_name"])
                    tools.append(function_tool(func))
        return tools

    def _load_handoffs(self):
        handoffs = []
        for handoff_config in self.config.get("handoffs", []):
            handoff_agent = Agent(
                name=handoff_config["name"],
                instructions=handoff_config["instructions"],
            )
            handoffs.append(handoff_agent)
        return handoffs

    def __call__(self, prompts: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(prompts, str):
            prompts = [prompts]
            single = True
        else:
            single = False

        from concurrent.futures import ThreadPoolExecutor

        def process_single(prompt):
            result = Runner.run_sync(self.agent, prompt)
            output = result.final_output

            if isinstance(output, dict):
                output = json.dumps(output)
            elif not isinstance(output, str):
                output = str(output)

            return output

        with ThreadPoolExecutor(max_workers=10) as executor:
            responses = list(executor.map(process_single, prompts))

        return responses[0] if single else responses
