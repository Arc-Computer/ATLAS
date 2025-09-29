# Wrapper Configs

| File | Use Case |
|------|----------|
| `openai_existing_agent.yaml` | Compatibility mode that wraps an existing OpenAI agent (Responses/Assistants). Ideal for GEPA optimization after the quick evaluation. |
| `openai_config.yaml` | Creates both teacher and student wrappers using OpenAI models (Responses API). Useful when you want ATLAS to instantiate both ends. |
| `your_agent.yaml` | Template for HTTP API integrations. |
| `python_agent.yaml` | Template for wrapping a Python function/agent. |

Wrapper types map to implementations in `wrappers/openai/` or `wrappers/custom_wrapper.py`.
