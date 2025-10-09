# ATLAS: A Continual Learning Framework for Production AI Agents

<div align="center">

<img src="public/ATLAS.png" alt="ATLAS Hero" width="900" style="border-radius: 12px;">

[![ATLAS-8B-Thinking](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ATLAS--8B--Thinking-blue)](https://huggingface.co/Arc-Intelligence/ATLAS-8B-Thinking) [![ATLAS-8B-Instruct](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ATLAS--8B--Instruct-blue)](https://huggingface.co/Arc-Intelligence/ATLAS-8B-Instruct) [![Arc-ATLAS-Teach Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Arc--ATLAS--Teach-green)](https://huggingface.co/datasets/Arc-Intelligence/Arc-ATLAS-Teach-v1) [![Docs](https://img.shields.io/badge/Docs-latest-green)](https://docs.arc.computer) [![PyPI version](https://img.shields.io/pypi/v/arc-atlas.svg)](https://pypi.org/project/arc-atlas/) [![Python 3.11 | 3.12](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue)](#installation)

</div>

ATLAS is an architecture for production teams that need AI agents to improve from user interactions and feedback *after* deployment. Everything is built around a closed-loop learning system—run the loop at inference time for real-time quality control, then feed the traces into online/offline training jobs when you’re ready to adapt or fine-tune.

## Why It Matters

- **Closed-loop runtime gains**: +15.7% average accuracy, +31% completion, 97% non-degradation, ~50% token savings
- **Online optimization**: Up to +165% domain-specific improvement in ~2 hours (no GPU required)
- **Offline reinforcement learning**: Long-horizon improvements with full control over checkpoints

Together, these capabilities help your agents compound expertise instead of relearning fixes manually.

## Four Components, One Loop

1. **Reasoning Core** – closed-loop runtime that pairs your target model with a reviewer
2. **Reward System (RIM)** – converts edits, approvals, and tool usage into dense reward signals
3. **Learning Engine** – online GEPA and offline GRPO for continuous adaptation
4. **Persistent Memory** – durable trace store for evaluation and retraining

<div align="center">
<img src="public/system-architecture.png" alt="ATLAS System Architecture Diagram" width="800" style="border-radius: 12px;">
<br>
<em>Figure: ATLAS keeps the agent in a learn–evaluate–update cycle.</em>
</div>

## Choose Your Path

| Goal | Best Starting Point | Key Docs |
|------|--------------------|----------|
| Orchestrate an existing agent with quality control | SDK Runtime Loop | [`SDK Quickstart`](https://docs.arc.computer/sdk/quickstart) |
| Adapt prompts to a new domain quickly | GEPA Online Optimization | [`Online Optimization`](https://docs.arc.computer/training/online/optimize-with-atlas) |
| Train custom checkpoints with RL | GRPO Offline Training | [`Full Training Walkthrough`](https://docs.arc.computer/first-experiment) |

To learn more, read about our production use cases ([Introducing ATLAS](https://www.arc.computer/blog/introducing-atlas), [ATLAS SRE Diagnosis](https://www.arc.computer/blog/atlas-sre-diagnosis)) and the research that underpins the framework, from our reward system ([ATLAS Reward System](https://www.arc.computer/blog/ATLAS-Reward-System)) to our online optimization results ([Supercharging RL with Online Optimization](https://www.arc.computer/blog/supercharging-rl-with-online-optimization)).

📄 **[Read the ATLAS Technical Report](https://docs.arc.computer/technical-report)** for methodology and benchmarks.

📚 **[Full Documentation](https://docs.arc.computer)** for comprehensive guides, API reference, and examples.

## Quickstart — Evaluate, Then Optimize

Start by measuring how much the closed-loop runtime (with GPT-5 acting as the reviewer) and the ATLAS Reward System improve one of your agents. Once you see the delta, graduate to the full GEPA optimization loop.

### Part A · 5 minutes — Score Baseline vs Teaching
1. Install the managed runtime (Python 3.11+):
   ```bash
   pip install arc-atlas
   ```
   <sub>If you want to run the example scripts from this repo, clone it and install the extras in editable mode: `git clone https://github.com/Arc-Computer/atlas-sdk && cd atlas-sdk && pip install -e .[dev]`.</sub>
2. Set credentials (OpenAI for models, Gemini for the reward judge):
   ```bash
   export OPENAI_API_KEY=sk-...
   export GEMINI_API_KEY=your_gemini_key
   ```
3. Run the quick evaluator. It captures a baseline response, asks the reviewer model (`--teacher-model`) for guidance, has the target model (`--student-model`) retry with that guidance, and scores both with `RIMReward`.
   ```bash
   python examples/quickstart/evaluate.py \
     --question "Masha braided her dolls' hair..." \
     --teacher-model gpt-5 \
     --student-model gpt-4o-mini
   ```
   > **Note:** The CLI flags map directly to the reviewer (`--teacher-model`) and target (`--student-model`) roles inside the closed-loop runtime.
   Example output:
   ```text
   ========================================================================
   Baseline student answer:
   ...
   Reward (baseline): 0.342
   Reward (with teaching): 0.781
   Delta: +0.439
   ========================================================================
   ```
   Change `--student-model` if you want to evaluate a different agent (any OpenAI Responses-compatible model).

### Part B · ~2 hours (optional) — Run GEPA Prompt Optimization
Reuse the proven compatibility config and let GEPA evolve the system prompts that steer the reviewer.

Before you start, make sure the same keys from Part A are exported—`OPENAI_API_KEY` for model calls and `GEMINI_API_KEY` for the reward judges.

```bash
# Optionally pin the models the script will call
export TEACHER_MODEL=gpt-5
export STUDENT_MODEL=gpt-5-mini

./scripts/openai_agent_atlas.sh configs/wrappers/openai_existing_agent.yaml
```
The config relies on the agents registry; edit `agents.target` inside the YAML to match your production connector.

You can also start from `configs/examples/quickstart.yaml`, which reuses the same wrapper with minimal overrides.

OpenAI currently limits Assistants to GPT-4.x models, so the GEPA wrapper defaults to `gpt-4.1` (or you can set `TEACHER_MODEL` to `Arc-Intelligence/ATLAS-8B-Thinking`).

This loop iterates up to 40 evaluations (≈$10 in API spend) and writes the best prompts to `optimized_prompts.json`. Attach those prompts to your agent once you're ready for deployment.

---

## How the Continual Loop Fits Together

Each quickstart mode taps into the same learning cycle:

1. **Evaluate (examples/quickstart/evaluate.py)** – capture baseline and reviewer-guided traces, then use the RIM judges to quantify the lift. Nothing feeds back yet; this is your observation step.
2. **Optimize (GEPA configs)** – reuse those reward deltas as fitness so the prompt optimizer keeps proposing better system prompts. Reviewer prompts change, target responses improve, rewards climb.
3. **Train (GRPO runs)** – when you need more than prompt tweaks, use the judged data to update the reviewer weights themselves. New data → judges score it → the reviewer adapts → refreshed prompts guide the target model → repeat.

That evaluate → optimize → train arc is the “self-improvement at scale” loop: fresh interactions enter RIM, the reward signal decides what to keep or discard, and ATLAS updates the learning components so the deployed agent never goes stale.

---

For a deeper tour of the codebase, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Implementation Paths

Each path maps to a component of the architecture, allowing you to adopt ATLAS incrementally.

### Path 1 — Adapt an Existing Agent in Hours

Use the **Learning Engine (GEPA)** and a pre-trained **Reasoning Core** to optimize your existing agent for a specific task. This is the most common starting point. See the [Online Optimization Guide](https://docs.arc.computer/training/online/optimize-with-atlas) for more.

- **Prerequisites**: An agent accessible via API, Python function, or OpenAI Assistant ID. API key for a reflection model (e.g., OpenAI, Gemini).
- **Command**:
  ```bash
  # Set API key for the reflection model
  export OPENAI_API_KEY="your-key-here"

  # This script wraps an existing agent and optimizes it
  ./scripts/openai_agent_atlas.sh configs/wrappers/openai_existing_agent.yaml
  ```
  Update `agents.target` in the config with your production connector before running the script.
- **Expected Outcome**: A set of optimized system prompts in `optimized_prompts.json`. This process takes ~2 hours and costs ~$10 in API fees, delivering a performance improvement of up to +165% (measured as reward delta on evaluation tasks). See [Supercharging RL with Online Optimization](https://www.arc.computer/blog/supercharging-rl-with-online-optimization) for experimental setup.

### Path 2 — Deploy a Standalone Reward System

Use the **Reward System (RIM)** to evaluate agent performance with state-of-the-art accuracy. This is useful for benchmarking or generating high-quality data for fine-tuning. Learn more in the [Reward System Documentation](https://docs.arc.computer/concepts/reward-design).

- **Prerequisites**: Python environment and API access for judge models (e.g., Gemini).
- **Code**:
  ```python
  from RIM.reward_adapter import RIMReward

  # Initialize the reward system from its config file
  reward_system = RIMReward(config_path='configs/rim_config.yaml')

  # Evaluate any interaction
  evaluation = reward_system.evaluate(prompt="<user_prompt>", response="<agent_response>")
  print(f"Score: {evaluation.score}, Rationale: {evaluation.rationale}")
  print(f"Per-judge scores: {evaluation.judge_scores}")
  ```
- **Expected Outcome**: A lightweight result object exposing the aggregated score, rationale, and per-judge details. Advanced callers can inspect `evaluation.extra["info"]` for the raw judge payload. For batched or high-throughput evaluation, call the `RIMReward` instance directly with lists of prompts and completions. The quickstart script prints aggregated and per-judge scores by default; pass `--verbose-judges` to surface full judge traces. The system achieves 93.7% accuracy on RewardBench V2, outperforming all public models. Benchmarks and judge configuration are detailed in [ATLAS Reward System](https://www.arc.computer/blog/atlas-reward-system).

### Path 3 — Train a Custom Reasoning Core

Use the full **Learning Engine (GRPO)** to train a new checkpoint on your own data. This is for advanced use cases requiring deep domain specialization. Follow the [Custom Training Guide](https://docs.arc.computer/first-experiment).

- **Prerequisites**: Multi-GPU environment (4-8x A100/H100 recommended).
- **Commands (Minimal Smoke Test)**:
  ```bash
  # Phase 1: SFT Warmup (1 epoch)
  scripts/launch.sh 1 configs/run/teacher_sft.yaml report_to=null save_final_model=false num_train_epochs=1

  # Phase 2: RL Training with vLLM (4 steps)
  scripts/launch_with_server.sh 1 1 configs/run/teacher_rcl.yaml report_to=null max_steps=4 eval_steps=1
  ```
  > The config filenames retain the historical `teacher_*` naming from the runtime stack—they produce the same reviewer checkpoint used in the closed-loop system.
- **Expected Outcome**: A custom-trained reasoning core checkpoint. The closed-loop runtime plus GRPO training delivers an average **+15.7%** accuracy lift, with optional GEPA optimization stacking on an additional **+165%** domain-specific gain in ~2 hours.

---

## Core Distinctions

<details>
<summary><strong>Why not just fine-tune?</strong></summary>

Fine-tuning (or RLHF) creates a static, updated version of a model. It is compute-intensive and risks catastrophic forgetting. When the world changes, you must repeat the entire process.

ATLAS creates a **dynamic, continual learning loop**. The closed-loop architecture separates foundational knowledge from task-specific adaptation. This means:
- **No Catastrophic Forgetting**: The student model's weights are never changed, preserving its original capabilities.
- **Rapid Adaptation**: The online learning loop adapts to new tasks in hours, not weeks.
- **Compounding Knowledge**: Skills learned from one task can be reapplied to others, creating a library of reusable "skill capsules."

</details>

<details>
<summary><strong>Production Notes</strong></summary>

- **Observability**: The system is designed for production monitoring. Log reward scores, KL divergence, and non-degradation rates to track performance. Integrate with Prometheus or Datadog.
- **Failure Modes**: The most common failure mode is a reward collapse during RL training. This is mitigated by tuning the `beta` (KL divergence) parameter in `teacher_grpo.yaml` to prevent the policy from deviating too far from its reference.
- **Hardware**: Offline training runs on 8xA100 (40GB) or equivalent. Online optimization is API-driven and requires no specialized hardware. Inference can run on a single 16GB GPU.

</details>

---

## Installation

**Quick install (runtime + exporter):**
```sh
pip install arc-atlas
```

Conda is recommended for environment management when developing locally. The repository has been validated with Python 3.11 and 3.12.

**Python 3.11:**
```sh
bash scripts/install_py311.sh
```

**Python 3.12:**
```sh
bash scripts/install_py312.sh
```
For detailed setup, see the [Installation Guide](https://docs.arc.computer/installation).

---

## Citation

If you use ATLAS in your research, please cite:
```bibtex
@article{atlas2025,
  title     = {ATLAS: A Hybrid RL Architecture for Compounding Intelligence},
  author    = {Arc Intelligence},
  journal   = {arXiv preprint},
  year      = {2025}
}
```
