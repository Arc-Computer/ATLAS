from __future__ import annotations

from typing import Any, Callable, Dict

from .providers import available as available_providers
from .providers import load as load_provider

BatchAgent = Callable[[Any], Any]


def load_agent(provider: str, params: Dict[str, Any]) -> BatchAgent:
    return load_provider(provider, params)


def load_wrapper(wrapper_type: str, config: Dict[str, Any]) -> BatchAgent:
    provider = _resolve_legacy_provider(wrapper_type, config)
    params = dict(config)
    params.pop("integration_type", None)
    return load_agent(provider, params)


def list_providers() -> list[str]:
    return available_providers()


def _resolve_legacy_provider(wrapper_type: str, config: Dict[str, Any]) -> str:
    if "." in wrapper_type:
        return wrapper_type
    mapping = {
        "openai_assistant": "openai.assistant",
        "openai_sdk": "openai.sdk",
    }
    if wrapper_type in mapping:
        return mapping[wrapper_type]
    if wrapper_type == "custom":
        integration = config.get("integration_type", "http_api")
        if integration == "python_function":
            return "python.callable"
        if integration == "cli_command":
            return "cli.command"
        return "http.api"
    raise ValueError(f"Unknown wrapper type: {wrapper_type}")
