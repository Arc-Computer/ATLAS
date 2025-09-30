from __future__ import annotations

import os
from typing import Any, Callable, Iterable, List, Union


PromptInput = Union[str, List[str]]
PromptOutput = Union[str, List[str]]
SingleGenerator = Callable[[str], str]
BatchGenerator = Callable[[PromptInput], PromptOutput]


def expand_env(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    return value


def as_batch_callable(generator: SingleGenerator) -> BatchGenerator:
    def _call(prompts: PromptInput) -> PromptOutput:
        if isinstance(prompts, str):
            return generator(prompts)
        return [generator(prompt) for prompt in prompts]

    return _call
