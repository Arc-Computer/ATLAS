# Atlas Core · Learning Engine for Adaptive Agents

<div align="center">

<img src="public/ATLAS.png" alt="ATLAS Hero" width="900" style="border-radius: 12px;" />

[![ATLAS-8B-Thinking](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ATLAS--8B--Thinking-blue)](https://huggingface.co/Arc-Intelligence/ATLAS-8B-Thinking)
[![ATLAS-8B-Instruct](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ATLAS--8B--Instruct-blue)](https://huggingface.co/Arc-Intelligence/ATLAS-8B-Instruct)
[![Arc-ATLAS-Teach Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Arc--ATLAS--Teach-green)](https://huggingface.co/datasets/Arc-Intelligence/Arc-ATLAS-Teach-v1)
[![Docs](https://img.shields.io/badge/Docs-latest-green)](https://docs.arc.computer)
[![Python 3.11 | 3.12](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue)](#installation)

</div>

# What is Atlas?

Atlas is the learning layer for production agents, giving them a way to adapt and learn in real time from every task. The runtime SDK (`atlas-sdk`) wraps any agent in an adaptive dual-agent reasoning loop (student + teacher) guided by reward signals that triages tasks, probes capability, and chooses the right supervision lane while streaming rich telemetry. This repository, **Atlas Core**, turns those adaptive episodes into new teacher checkpoints with offline GRPO training and provides the reward tooling that powers both halves.

| When you need… | Use | Highlights |
|----------------|-----|------------|
| Runtime continual learning: triage → probe → lane routing, telemetry, JSONL export | [`Arc-Computer/atlas-sdk`](https://github.com/Arc-Computer/atlas-sdk) | Drop-in runtime harness with a verifying teacher, storage helpers (`atlas init`), export/review CLI (`arc-atlas`, `arc-atlas review`), persona telemetry |
| Offline optimization: GRPO training, reward adapters, analysis utilities | `Arc-Computer/ATLAS` (this repo) | GRPO trainer, RIM reward system, data loaders, launch scripts |

<div align="center">
  <img src="public/runtime-2.png" alt="Atlas SDK adaptive runtime flow diagram showing triage, capability probe, and lane routing (auto, paired, coach, escalate)" width="900" />
  <p><em>The runtime SDK triages each task, probes capability, and routes it into the right supervision lane before your agent executes.</em></p>
</div>

## Architecture at a Glance

<div align="center">
  <img src="docs/images/system-architecture.png" alt="Atlas architecture showing reasoning core, reward system, learning engine, and persistent memory connected to agent frameworks" width="900" style="border-radius: 12px;" />
  <p><em>Runtime episodes feed the reward system and persistent memory; Atlas Core consumes those exports to train new teachers with GRPO.</em></p>
</div>

- **Reasoning Core** – Student and verifying teacher personas live in `atlas-sdk`, wrapping your agent with adaptive supervision in lanes (`auto`, `paired`, `coach`, `escalate`).
- **Reward System (RIM)** – Shared evaluators score every step and session; the same judges label runtime telemetry and training data.
- **Learning Engine** – This repo’s GRPO trainer ingests exported traces (`arc-atlas`) to produce updated teacher checkpoints.
- **Persistent Memory** – Postgres + JSONL exports capture triage dossiers, adaptive summaries, persona updates, and reward payloads so the learning loop compounds over time.

## End-to-End Workflow

1. **Wrap your agent with the runtime SDK** – Follow the [`SDK Quickstart`](https://docs.arc.computer/sdk/quickstart) to install `arc-atlas`, run discovery with `atlas env init`, and execute tasks via `atlas run` or `atlas.core.run`. Each task is triaged, probed, and routed into `auto`, `paired`, `coach`, or `escalate`, while adaptive summaries, persona updates, and rewards are recorded.
2. **Persist and export telemetry** – Provision Postgres with `atlas init` (optional helper) and export sessions with `arc-atlas --database-url … --include-status approved --output traces.jsonl`. Before exporting new data, review and approve pending sessions using `arc-atlas review` so only trusted traces feed training. Every record carries the triage dossier, lane decision, probe confidence, guidance, reward breakdowns, and review status.
3. **Train offline with Atlas Core** – Use this repository’s GRPO pipeline (`python scripts/run_offline_pipeline.py --export-path <traces.jsonl>`) to turn runtime traces into a new teacher checkpoint. Redeploy the checkpoint back through the SDK to close the loop.

This hand-off keeps the learning flywheel tight: runtime captures adaptive behaviour, exports become training data, and Atlas Core ships the update back into production.

## What's in this Repository

- `scripts/run_offline_pipeline.py` – One-command launcher that composes Hydra configs, spins up GRPO training, and handles logging/output directories.
- `configs/` – Hydra library for models, datasets (`runtime_traces.yaml`), trainers (`teacher_grpo.yaml`), and ready-to-run recipes (e.g., `configs/run/teacher_sft.yaml`, `configs/run/teacher_grpo.yaml`).
- `RIM/` – Reward adapters and judge definitions shared between runtime evaluation and offline training.
- `train.py` / `trainers/` – Core GRPO loop, data loaders, and evaluation utilities.
- `docs/` – Mintlify docs (synced with [docs.arc.computer](https://docs.arc.computer)) covering runtime orchestration and training deep dives.

## Configuration Overview

Hydra bundles Atlas defaults into composable groups:

- `model@_global_` → `configs/model/`
- `data@_global_` → `configs/data/`
- `trainer@_global_` → `configs/trainer/`

Starter configs ship in:

- `configs/examples/quickstart.yaml` (minimal overrides for the helper script)
- `configs/run/teacher_rcl.yaml` (reward conditioned learning recipe for GRPO training)

Deep dives and override recipes live in the [Training Configuration guide](https://docs.arc.computer/training/configuration).

## Offline Quickstart

1. **Prepare the runtime export**
   ```bash
   # From the atlas-sdk repo after running adaptive episodes
   atlas init  # optional helper to launch Postgres
   arc-atlas review sessions --database-url postgresql://atlas:atlas@localhost:5433/atlas --status pending
   # Approve or quarantine as needed, then export approved sessions
   arc-atlas --database-url postgresql://atlas:atlas@localhost:5433/atlas \
     --include-status approved \
     --output traces/runtime.jsonl
   ```
   Each record carries `triage_dossier`, `adaptive_summary`, persona usage/updates, plan/step traces, and reward payloads—the exact inputs Atlas Core expects.

2. **Launch GRPO training**
   ```bash
   python scripts/run_offline_pipeline.py \
     --export-path traces/runtime.jsonl \
     output_dir=results/teacher-grpo
   ```
   Override Hydra arguments (model, batch size, GPUs) as needed; the helper wires up `configs/run/teacher_rcl.yaml` by default.

3. **Redeploy the checkpoint**
   Point the runtime SDK at `results/teacher-grpo/rl_checkpoint/` (or your chosen output dir) to load the new teacher, then rerun `atlas.core.run` to close the loop.

## GKD Distillation (On-Policy Knowledge Transfer)

For faster, more compute-efficient training, use **Generalized Knowledge Distillation (GKD)** to distill Atlas runtime traces into smaller student models. GKD is 9-30× faster than GRPO while preserving learning quality.

### Quick Start

```bash
# Train a 7B student model distilled from 14B teacher
python train.py \
  --config-name teacher_gkd \
  teacher_model_name_or_path=Qwen/Qwen2.5-14B-Instruct \
  model.model_name_or_path=Qwen/Qwen2.5-7B-Instruct \
  trainer.min_reward=0.8
```

### Key Benefits

- **9-30× faster** than GRPO training
- **Direct Postgres access** (no JSONL export needed)
- **baseline comparison metrics** built-in (success delta, token efficiency)
- **Multi-turn native** for conversation workflows

### When to Use GKD vs GRPO

| Use GKD when... | Use GRPO when... |
|----------------|------------------|
| You have Atlas traces and want to distill into a smaller model | You need to train a new policy from scratch |
| You need fast iteration (hours vs days) | You need interactive RL with environment feedback |
| You want to deploy production-ready distilled models | You're exploring new reward structures |

See the [GKD Training Guide](https://docs.arc.computer/training/offline/gkd-training) for configuration options, hyperparameter tuning, and troubleshooting.

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
- [SDK Docs](https://docs.arc.computer/sdk/quickstart) – Runtime orchestration, export/review CLI, online adaptation
- [Evaluation Harnesses](https://docs.arc.computer/benchmarks/evaluation-harnesses) – Learning, runtime, and reward harness workflows
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
