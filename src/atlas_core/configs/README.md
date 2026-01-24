# Config Directory Overview

| Folder / File            | Description |
|--------------------------|-------------|
| `data/`                  | Dataset descriptors and loaders (runtime traces, Arc-ATLAS RL/SFT, benchmarks). |
| `model/`                 | Model definitions and training checkpoints. |
| `trainer/`               | Trainer-specific settings (GRPO, GKD, SFT). |
| `reward/`                | Reward presets (RIM-based, MetaMath binary, teaching templates). |
| `recipe/`                | Full experiment bundles (quickstart, teacher GRPO/SFT/GKD). |
| `integrations/`          | Provider integration configs (HTTP, Python callable, OpenAI). |
| `reward_system/`         | Reward Interpretation System (RIM) configs used by the reward adapter. |
