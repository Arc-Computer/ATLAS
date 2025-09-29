# Wrapper Configs

| File | Use Case |
|------|----------|
| `openai_existing_agent.yaml` | Compatibility mode that wraps an existing OpenAI agent (Responses/Assistants). Ideal for GEPA optimization after the quick evaluation. |
| `openai_config.yaml` | Creates both teacher and student wrappers using OpenAI models (Responses API). Useful when you want ATLAS to instantiate both ends. |

To integrate a different agent, copy `openai_existing_agent.yaml` (e.g., `cp configs/wrappers/openai_existing_agent.yaml configs/wrappers/your_agent.yaml`) and adjust the `user_agent` block. The wrappers support HTTP APIs, Python functions, and OpenAI Assistants—see the docs for concrete examples.

Wrapper types map to implementations in `wrappers/openai/` or `wrappers/custom_wrapper.py`.
