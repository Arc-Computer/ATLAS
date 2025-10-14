# ATLAS Core · Offline GRPO Training and Reward Tooling

<div align="center">

<img src="public/ATLAS.png" alt="ATLAS Hero" width="900" style="border-radius: 12px;">

[![ATLAS-8B-Thinking](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ATLAS--8B--Thinking-blue)](https://huggingface.co/Arc-Intelligence/ATLAS-8B-Thinking)
[![ATLAS-8B-Instruct](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ATLAS--8B--Instruct-blue)](https://huggingface.co/Arc-Intelligence/ATLAS-8B-Instruct)
[![Arc-ATLAS-Teach Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Arc--ATLAS--Teach-green)](https://huggingface.co/datasets/Arc-Intelligence/Arc-ATLAS-Teach-v1)
[![Docs](https://img.shields.io/badge/Docs-latest-green)](https://docs.arc.computer)
[![Python 3.11 | 3.12](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue)](#installation)

</div>

Atlas Core is the **offline** side of ATLAS: GRPO training loops, reward tooling, and utilities for turning runtime traces into deployable teacher checkpoints. The **atlas-sdk** repository now owns the runtime, telemetry, and online continual learning story. Think of the split as:

| What you need | Repo | Highlights |
|---------------|------|------------|
| Runtime quality control, trace export, online adaptation | [`Arc-Computer/atlas-sdk`](https://github.com/Arc-Computer/atlas-sdk) | Bring-your-own-agent orchestration, telemetry, continual learning CLI |
| Offline GRPO training and reward tooling | `Arc-Computer/ATLAS` (this repo) | Train teacher checkpoints from exported traces, analyze rewards, manage checkpoints |

By keeping online optimization inside the SDK and offline RL here, teams get a clean hand-off: **export runtime traces → run GRPO → deploy the new teacher.**

## Why the Offline Stack Matters

- **Production-grounded data** – You own the traces captured in production. Atlas Core consumes them directly without relabeling.
- **Repeatable GRPO pipeline** – Hydrated configs, reward adapters, and vLLM integration ship a proven recipe for training teacher checkpoints.
- **Separation of concerns** – Online continual learning, real-time adaptation, and agent wrappers live in the SDK. Atlas Core focuses purely on training.

<div align="center">
<img src="public/system-architecture.png" alt="ATLAS System Architecture Diagram" width="800" style="border-radius: 12px;">
<br>
<em>Offline GRPO sits downstream of the runtime loop: export traces → train → redeploy.</em>
</div>

## Workflow at a Glance

Atlas Core fits into a simple, repeatable loop:

1. **Export traces** with the SDK runtime (`arc-atlas --database-url … --output traces.jsonl`). The SDK owns online orchestration, telemetry, and JSONL export.
2. **Launch offline training** with `python scripts/run_offline_pipeline.py --export-path <traces.jsonl>`. The helper applies the correct Hydra overrides so you can scale training without hand-written commands.
3. **Evaluate and redeploy** the new teacher checkpoint, updating runtime configs in the SDK when you are satisfied with the lift.

Need the hands-on version? Follow the [Quickstart tutorial](https://docs.arc.computer/quickstart) for the complete step-by-step instructions.

## Configuration Overview

Hydra bundles Atlas defaults into composable groups:

- `model@_global_` → `configs/model/`
- `data@_global_` → `configs/data/`
- `trainer@_global_` → `configs/trainer/`

Starter configs ship in:

- `configs/examples/quickstart.yaml` (minimal overrides for the helper script)
- `configs/demo/runtime_grpo.yaml` (documented walkthrough used in the Mintlify docs)

Deep dives and override recipes live in the [Training Configuration guide](https://docs.arc.computer/training/configuration).

## Rewards Only?

Need scoring without training? Import `RIMReward` directly:

```python
from RIM.reward_adapter import RIMReward

reward_system = RIMReward(config_path="configs/rim_config.yaml")
score = reward_system.evaluate(prompt="...", response="...")
print(score.score, score.rationale)
```

## Documentation & Resources

- [Atlas Core Docs](https://docs.arc.computer) – Offline training guides, reward system reference, architecture deep dives
- [SDK Docs](https://docs.arc.computer/sdk/quickstart) – Runtime orchestration, export CLI, online adaptation
- [Technical Report](https://docs.arc.computer/reference/technical-report) – Research, benchmarks, and methodology

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-py312.txt
```

> Need GPU-backed training? Install PyTorch matching your CUDA stack, then run `pip install -r requirements-py312.txt`.
>
> On Linux/CUDA environments the pinned `bitsandbytes` wheel will install automatically; on macOS or Windows it is skipped.

## Development

- Format / lint: `ruff check .`
- Tests: `pytest`
- Docs sanity: `mintlify broken-links` (requires interactive prompt today)
- Type checking: `pyright` (covers `train.py`, offline CLI helpers, and the runtime trace ingest path; see `pyrightconfig.json`)

We track major changes in `CHANGELOG.md`.

## License

MIT © Arc Computer
