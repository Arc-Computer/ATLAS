from __future__ import annotations

from typing import Any, Callable, Dict

from .base import BatchGenerator

ProviderLoader = Callable[[Dict[str, Any]], BatchGenerator]
_registry: Dict[str, ProviderLoader] = {}


def register(name: str, loader: ProviderLoader) -> None:
    _registry[name] = loader


def load(name: str, params: Dict[str, Any]):
    if name not in _registry:
        raise ValueError(f"Unknown provider: {name}")
    return _registry[name](params)


def available() -> list[str]:
    return sorted(_registry.keys())


from . import openai_assistant, openai_sdk, http_api, python_callable, cli  # noqa: E402,F401

register("openai.assistant", openai_assistant.load)
register("openai.sdk", openai_sdk.load)
register("http.api", http_api.load)
register("python.callable", python_callable.load)
register("cli.command", cli.load)
