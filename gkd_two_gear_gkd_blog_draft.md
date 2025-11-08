# Two Gears of On-Policy Distillation from Atlas SDK Traces

Atlas is a system for continual learning from agent workflows, designed to close the loop between real-world agent execution and model improvement. The platform is composed of two parts: the **Atlas SDK**, a runtime component that captures rich causality data from agent interactions, and **Atlas Core**, an offline training engine that uses this data to build better models.

Atlas is built for teams who want continual learning directly from runtime experience. The SDK captures causality traces (student attempt → teacher intervention → outcome) in production; Atlas Core turns those traces into new checkpoints without building a fresh data pipeline. RLVR, self-play, and LoRA adapters are powerful, but they all require bespoke reward wiring or infrastructure work. We wanted to document what happens when you stay entirely inside the trace stream and use on-policy distillation as the first lever in your roadmap.

This note walks through two Generalized Knowledge Distillation (GKD) runs on the same GSM8K dataset exported from the SDK and stored in Postgres. Both use stock Atlas tooling (Hydra configs + `scripts/validate_gkd.py`) and run on a single NVIDIA DGX Spark. The only difference is the operating point:

- **Gear 1 — Fast Sweep**: Default config for quick plumbing checks and rapid iteration.
- **Gear 2 — High-Quality Sweep**: Tighter reward filters and conservative generation/optimizer settings for higher reliability targets.

We’ll fill in the final telemetry (loss curves, Baseline Comparison metrics, token deltas) once the runs complete; the rest of the workflow and reasoning is documented here so other researchers can reproduce it.

---

## Why Two Gears?

A managed continual learning service needs to balance two competing demands: the need for rapid, iterative validation and the push for production-grade reliability. How do you quickly check if a new batch of production traces has broken the pipeline? And how do you schedule the deeper, more expensive runs for workflows that demand 95%+ success rates?

This requires a system with multiple operating points. Instead of treating distillation as a monolithic process, we define two "gears" that use the exact same training pipeline but with different Hydra configurations. This allows a research team to shift between a fast, diagnostic cadence and a high-quality, reliability-focused cadence without ever leaving the Atlas framework. One gear is for plumbing checks and directional feedback; the other is for hitting SLAs.

---

## Experiment Setup

Our goal is to isolate the impact of the configuration, so both runs share an identical foundation.

- **Hardware**: Single DGX Spark (using the `nvcr.io/nvidia/pytorch:25.09-py3` container).
- **Models**: A `Qwen/Qwen2.5-14B-Instruct` teacher and a `Qwen/Qwen2.5-7B-Instruct` student.
- **Data**: A GSM8K dataset of 8,792 traces captured via the Atlas SDK, with 7,473 used for training and 1,319 for evaluation. The data is streamed directly from Postgres using the runtime storage client.
- **Trainer**: The stock `AtlasGKDTrainer` (`trainer=gkd` in Hydra), invoked via the `scripts/validate_gkd.py` validation script.
- **Metrics**: We track `train/loss`, `grad_norm`, and `eval/loss`, alongside wall-clock time and the Baseline Comparison telemetry (success delta and token reduction) logged to WandB.

---

## Gear 1 · The Fast Sweep

What can a 12-hour run on a single DGX tell us? The primary goal of the fast gear is to confirm that the end-to-end loop—from SDK trace capture to GKD training—is sound. It’s a sanity check, designed to provide directional telemetry and catch regressions before committing to a multi-day run.

**Config Highlights:**
- `lmbda=1.0`, `beta=0.5`
- `temperature=0.9`
- `min_reward=0.8`
- `learning_rate=2e-5`
- `max_steps=500`

The run finished in 11 hours and 45 minutes, providing a clear picture of the training dynamics.

#### Reading the Telemetry

The logs tell a story of a healthy and efficient run. The training loss dropped from an initial **0.0676** to **0.0294** in just 500 steps. More importantly, the evaluation loss tracked this descent, moving from **0.0437** down to **0.0394** across three checkpoints.

*[Image: A WandB chart showing train/loss and eval/loss for the Gear 1 fast sweep. The two lines should track each other closely, showing a steady downward trend.]*

This convergence is exactly what we want to see. The lack of divergence between the training and evaluation loss curves indicates that the student model is generalizing well from the teacher's guidance, not just overfitting to the training batch.

We also monitor gradient norms as a proxy for stability. Throughout the run, `grad_norm` remained bounded between **0.88** and **1.70**.

*[Image: A WandB chart showing grad_norm over 500 steps for the Gear 1 fast sweep. The line should be noisy but hover within a stable horizontal band.]*

Stable gradients, combined with a smooth loss decay, give us high confidence in the run's integrity. This is the "green light" we look for: a clear signal that the data is learnable and the configuration is sound, justifying an investment in the more expensive Gear 2 sweep.

---

## Gear 2 · The High-Quality Sweep

Where the fast gear prioritizes speed, the high-quality gear prioritizes reliability. The goal here is to push the student model toward its performance ceiling, creating a checkpoint that could meet a production SLA.

Our hypothesis is that by being more selective with our data and more conservative with our generation and optimization, we can distill a more robust policy.

**Planned Config:**
- `lmbda=1.0`, `beta=0.5`
- `temperature=0.6` (more conservative generation)
- `max_new_tokens=128` (tighter, more focused responses)
- `min_reward=0.9` (filters for only the highest-quality traces)
- `learning_rate` (a lower `3e-6`)
- An extended schedule (`num_train_epochs=5` or an equivalent `max_steps`).

**Expected Telemetry (Upcoming):**
- Training and evaluation loss trajectories under the stricter data filter.
- Gradient norms, which we expect to remain low and stable.
- Baseline Comparison deltas, hopefully showing improved success rates and tighter token budgets.
- Wall-clock time, which will serve as a benchmark for the cost of a reliability-focused run.

This is the gear we anchor our managed continual learning service to. Because the trainer and dataset plumbing are identical to the fast sweep, a team can schedule this deeper run weekly or biweekly just by submitting a different set of Hydra overrides.

---

## Comparing the Cadences

Once the high-quality run completes, we will compare the metrics side-by-side. The goal is not to declare a "winner," but to illustrate the trade-offs and demonstrate how a team can strategically alternate between gears.

| Metric | Gear 1 (Fast) | Gear 2 (High-Quality) |
| --- | --- | --- |
| **Train Loss (start → end)** | 0.0676 → 0.0294 | _Fill in_ |
| **Eval Loss (start → end)** | 0.0437 → 0.0394 | _Fill in_ |
| **Success Delta (vs. Baseline)** | _Fill in_ | _Fill in_ |
| **Token Reduction (%)** | _Fill in_ | _Fill in_ |
| **Wall-Clock (hrs)** | 11.75 | _Fill in_ |

In practice, a team might run Gear 1 after every new batch of traces is ingested to get a quick signal. If the telemetry is positive, they can schedule a Gear 2 run for the critical workflows where reliability and cost-efficiency are paramount. This two-gear system provides a practical framework for managing the cadence of continual learning.

---

## Where RLVR and Self-Play Fit

It's important to clarify that on-policy distillation isn't a replacement for all other learning methods. We still plan to publish a GRPO (RLVR-style) baseline on the exact same GSM8K data slice. So, when should a team choose one over the other?

Our perspective is this: GKD is the workhorse. It's the default choice when you have a trustworthy teacher and a stream of on-policy data from the Atlas SDK. It's computationally cheaper and provides a dense, token-level reward signal.

RLVR and self-play are specialist tools. They become invaluable in two primary scenarios:
1.  When you **lack a reliable teacher policy** and must discover behavior from scratch.
2.  When the task has a **sparse reward** (e.g., a game-winner or a multi-step task with only a final pass/fail), making token-level distillation impossible.

By establishing a robust two-gear GKD loop first, we create a reliable default for the 80% of cases, reserving the higher cost and complexity of RLVR for the 20% of problems that truly demand exploration.

---

## Methods Appendix

**Fast Sweep Command**

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 \
PYTHONPATH=. \
CUDA_VISIBLE_DEVICES=0 \
python scripts/validate_gkd.py \
  --student Qwen/Qwen2.5-7B-Instruct \
  --teacher Qwen/Qwen2.5-14B-Instruct \
  --dataset-name gsm8k \
  --dataset-config main \
  --dataset-max-samples 8792 \
  --train-limit 7473 \
  --eval-limit 1319 \
  --max-steps 500 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-5 \
  --lmbda 1.0 \
  --beta 0.5 \
  --temperature 0.9 \
  --max-new-tokens 256 \
  --eval-sample-size 256 \
  --min-reward 0.8 \
  --bf16
```

**High-Quality Sweep Command (Planned)**

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 \
PYTHONPATH=. \
CUDA_VISIBLE_DEVICES=0 \
python scripts/validate_gkd.py \
  --student Qwen/Qwen2.5-7B-Instruct \
  --teacher Qwen/Qwen2.5-14B-Instruct \
  --dataset-name gsm8k \
  --dataset-config main \
  --dataset-max-samples 8792 \
  --train-limit 7473 \
  --eval-limit 1319 \
  --num-train-epochs 5 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --learning-rate 3e-6 \
  --lmbda 1.0 \
  --beta 0.5 \
  --temperature 0.6 \
  --max-new-tokens 128 \
  --eval-sample-size 256 \
  --min-reward 0.9 \
  --bf16
```

Both scripts log to Weights & Biases and write metrics under `outputs/gkd_*`.

---

## Closing Thoughts

On-policy distillation provides a powerful and compute-efficient path for converting runtime experience into model improvements. By framing it as a two-gear system, we move from ad-hoc training runs to a repeatable engineering discipline. A fast gear for validation and a high-quality gear for reliability lets teams manage a continual learning cadence that is both responsive and robust.

This approach makes dense, per-token guidance the default, reserving the higher overhead of RLVR for problems where it's truly necessary. As we await the final metrics from the high-quality sweep, we hope this research log provides a useful template for other teams exploring how to structure their own learning loops.
