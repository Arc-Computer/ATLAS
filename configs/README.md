# Config Directory Overview

| Folder / File            | Description |
|--------------------------|-------------|
| `data/`                  | Dataset descriptors and loaders (arc-atlas RL/SFT, benchmarks). |
| `demo/`                  | End-to-end demo configurations (ITBench, open-source sandboxes). |
| `examples/`              | Ready-to-run configs referenced in docs (e.g., quickstart). |
| `model/`                 | Model definitions and training checkpoints. |
| `optimize/`              | GEPA/online optimization recipes. |
| `run/`                   | Hydra run configs for training/experiments. |
| `trainer/`               | Trainer-specific settings (reward, grpo, etc.). |
| `wrappers/`              | Agent integration configs (HTTP, Python fn, OpenAI responses/assistants). |
| `rim_config.yaml`        | Default reward-system configuration for quick evaluation. |
| `rim_offline_config.yaml`| RIM offline evaluation profile. |
