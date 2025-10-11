# Config Directory Overview

| Folder / File            | Description |
|--------------------------|-------------|
| `data/`                  | Dataset descriptors and loaders (runtime traces, arc-atlas RL/SFT, benchmarks). |
| `demo/`                  | End-to-end GRPO walkthrough configs that assume exported runtime traces. |
| `examples/`              | Ready-to-run configs referenced in docs (e.g., offline quickstart). |
| `model/`                 | Model definitions and training checkpoints. |
| `run/`                   | Hydra run configs for training/experiments. |
| `trainer/`               | Trainer-specific settings (GRPO, reward shaping, etc.). |
| `wrappers/`              | Agent integration configs (HTTP, Python fn, OpenAI responses/assistants). |
| `rim_config.yaml`        | Default reward-system configuration for quick evaluation. |
| `rim_offline_config.yaml`| RIM offline evaluation profile. |
