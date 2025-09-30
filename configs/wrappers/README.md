# Wrapper Configs

| File | Use Case |
|------|----------|
| `openai_existing_agent.yaml` | Compatibility harness for an existing production agent defined through the `agents.target` block. |
| `openai_config.yaml` | Runs ATLAS with both teacher and student hosted through OpenAI Assistants using the `agents` schema. |

All wrapper configs share the `agents:` map. Each entry specifies a `provider` and a `params` dictionary that plugs into the wrapper registry located at `wrappers/providers/`.

Built-in providers:
- `openai.assistant`
- `openai.sdk`
- `http.api`
- `python.callable`
- `cli.command`

Copy one of the configs, adjust the providers you need, and point the optimization script at the new file.
