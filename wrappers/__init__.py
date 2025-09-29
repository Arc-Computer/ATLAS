from typing import Any, Callable, Dict

from .openai import load_openai_wrapper


def load_wrapper(wrapper_type: str, config: Dict[str, Any]) -> Callable:
    if wrapper_type == "openai_sdk":
        return load_openai_wrapper("sdk", config)
    if wrapper_type == "openai_assistant":
        return load_openai_wrapper("assistant", config)
    if wrapper_type == "custom":
        from .custom_wrapper import CustomWrapper
        return CustomWrapper(config)
    raise ValueError(f"Unknown wrapper type: {wrapper_type}")
