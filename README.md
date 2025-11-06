# Atlas Core · Learning Engine for Adaptive Agents

<div align="center">

<img src="public/ATLAS.png" alt="ATLAS Hero" width="900" style="border-radius: 12px;" />

[![arXiv](https://img.shields.io/badge/arXiv-2511.01093-b31b1b.svg)](https://arxiv.org/abs/2511.01093)
[![ATLAS-8B-Thinking](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ATLAS--8B--Thinking-blue)](https://huggingface.co/Arc-Intelligence/ATLAS-8B-Thinking)
[![ATLAS-8B-Instruct](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ATLAS--8B--Instruct-blue)](https://huggingface.co/Arc-Intelligence/ATLAS-8B-Instruct)
[![Arc-ATLAS-Teach Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Arc--ATLAS--Teach-green)](https://huggingface.co/datasets/Arc-Intelligence/Arc-ATLAS-Teach-v1)
[![Docs](https://img.shields.io/badge/Docs-latest-green)](https://docs.arc.computer)
[![Python 3.11 | 3.12](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue)](#installation)

</div>

# What is Atlas?

Atlas is the learning layer for production agents, enabling continual learning from complex, high-stakes workflows. For teams pushing beyond context engineering into reinforcement learning and adaptive systems, Atlas captures the causality data needed to improve models from real-world agent execution.

The **Atlas SDK** wraps existing agent systems in a dual-agent reasoning loop without requiring codebase refactoring. It automatically discovers your agent configuration, routes supervision dynamically across lanes (auto, paired, coach), and captures causality traces (student attempt → teacher intervention → outcome) while streaming rich telemetry to Postgres.

**Atlas Core** (this repository) handles offline training with GRPO and on-policy distillation (GKD), sharing reward adapters with the runtime to train updated checkpoints for the teacher model from the causality data captured by the SDK. One-command model training that learns from your agent's actual execution patterns.

## Architecture at a Glance

<div align="center">
  <img src="docs/images/system-architecture.png" alt="Atlas architecture showing reasoning core, reward system, learning engine, and persistent memory connected to agent frameworks" width="900" style="border-radius: 12px;" />
  <p><em>The SDK captures causality traces and feeds the reward system; Atlas Core trains new teacher checkpoints from this data.</em></p>
</div>

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

2. **Train your model**
   Use this repository's training pipeline to update your teacher model from runtime traces. Atlas Core supports multiple training methods depending on your needs—see [Training Methods](https://docs.arc.computer/training) for detailed guidance.

   ```bash
   # Example: GRPO training
   python scripts/run_offline_pipeline.py \
     --export-path traces/runtime.jsonl \
     output_dir=results/teacher-grpo
   ```
   Override Hydra arguments (model, batch size, GPUs) as needed; the helper wires up `configs/run/teacher_rcl.yaml` by default.

3. **Redeploy the checkpoint**
   Point the runtime SDK at your output directory (e.g., `results/teacher-grpo/rl_checkpoint/`) to load the new teacher, then rerun `atlas.core.run` to close the loop.

## Training Methods

Atlas Core provides flexible training capabilities for different scenarios:

- **GRPO** – Reinforcement learning from reward signals in runtime traces. Updates teacher policies by optimizing for task success and efficiency.
- **GKD** – Distill large models into smaller, deployment-optimized variants. 9-30× faster training than GRPO for creating compact production models.
- **SFT** – Supervised fine-tuning on approved traces. Direct imitation learning from high-quality runtime episodes.

Each method uses the same Postgres-backed dataset infrastructure and Hydra configuration system. All training methods support direct database access, reward filtering, and multi-turn conversation workflows.

See the [Training Guide](https://docs.arc.computer/training) for detailed comparisons, configuration options, and when to use each method.

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
