from typing import Any, Dict

from .assistant import OpenAIAssistantWrapper
from .sdk import OpenAISDKWrapper

__all__ = ["OpenAIAssistantWrapper", "OpenAISDKWrapper"]

def load_openai_wrapper(wrapper_type: str, config: Dict[str, Any]):
    if wrapper_type == "assistant":
        return OpenAIAssistantWrapper(config)
    if wrapper_type == "sdk":
        return OpenAISDKWrapper(config)
    raise ValueError(f"Unknown OpenAI wrapper variant: {wrapper_type}")
