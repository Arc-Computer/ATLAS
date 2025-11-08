# Two Gears of On-Policy Distillation from Atlas SDK Traces

Atlas is a system for continual learning from agent workflows, designed to close the loop between real-world agent execution and model improvement. The platform is composed of two parts: the **Atlas SDK**, a runtime component that captures rich causality data from agent interactions, and **Atlas Core**, an offline training engine that uses this data to build better models.

Atlas is built for teams who want continual learning directly from runtime experience. The SDK captures causality traces (student attempt → teacher intervention → outcome) in production; Atlas Core turns those traces into new checkpoints without building a fresh data pipeline. RLVR, self-play, and LoRA adapters are powerful, but they all require bespoke reward wiring or infrastructure work. We wanted to document what happens when you stay entirely inside the trace stream and use on-policy distillation as the first lever in your roadmap.

This note walks through two Generalized Knowledge Distillation (GKD) runs on the same GSM8K dataset exported from the SDK and stored in Postgres. Both use stock Atlas tooling (Hydra configs + `scripts/validate_gkd.py`) and run on a single NVIDIA DGX Spark. The only difference is the operating point:

- **Gear 1 — Fast Sweep**: Default config for quick plumbing checks and rapid iteration.
- **Gear 2 — High-Quality Sweep**: Tighter reward filters and conservative generation/optimizer settings for higher reliability targets.

We’ll fill in the final telemetry (loss curves, Baseline Comparison metrics, token deltas) once the runs complete; the rest of the workflow and reasoning is documented here so other researchers can reproduce it.

---

## Why two gears?

Managed continual learning (our internal service definition) promises recurring distillation cadences without handing customers an ML ops project. In practice that means giving practitioners a way to:

1. Validate the SDK → Postgres → GKD loop quickly after each batch of traces.
2. Schedule deeper sweeps on the workflows that need 95%+ reliability.

You should be able to do both without touching the codepath. Atlas already exposes the knobs through Hydra, so we treat “fast” and “high-quality” as two presets on the same trainer.

---

## Experiment Setup

- **Hardware**: Single DGX Spark (NVIDIA PyTorch container `nvcr.io/nvidia/pytorch:25.09-py3`).
- **Models**: Teacher `Qwen/Qwen2.5-14B-Instruct`, student `Qwen/Qwen2.5-7B-Instruct`.
- **Data**: GSM8K traces captured via Atlas SDK, streamed from Postgres using the runtime storage client.
- **Trainer**: `AtlasGKDTrainer` (`trainer=gkd` in Hydra) invoked through `scripts/validate_gkd.py`.
- **Metrics**: `train/loss`, `grad_norm`, `eval/loss`, Baseline Comparison telemetry (success delta, token reduction), wall-clock / GPU hours.

Both runs use the same export (`dataset-max-samples=8792`, `train-limit=7473`, `eval-limit=1319`) so differences come solely from configuration.

---

## Gear 1 · Fast GKD Sweep

Purpose: confirm the end-to-end loop works on fresh traces and get directional telemetry within a single afternoon.

**Config highlights**

- `lmbda=1.0`, `beta=0.5`
- `temperature=0.9`
- `max_new_tokens=256`
- `min_reward=0.8`
- Learning rate `2e-5`
- `max_steps=500`, `per_device_train_batch_size=2`, `gradient_accumulation_steps=4`

**Telemetry (to be filled in)**

- Training loss trajectory (expected: 0.0676 → 0.0294 by epoch ≈0.21)
- Eval loss checkpoints (expected: ~0.0437, ~0.0397, …)
- Gradient norms (expected range: 0.88–1.70)
- Baseline Comparison deltas (success + token savings)
- Wall-clock / GPU hours on DGX Spark

Interpretation: when the loss curve drops this quickly and gradients stay in band, you know the trace ingest, Hydra config, and TRL wiring are behaving. This “gear” is what we run after every batch of runtime traces before deciding whether to iterate further.

---

## Gear 2 · High-Quality GKD Sweep

Purpose: push toward the ≥95% reliability tier by biasing toward top-quality traces and constraining generation.

**Config highlights**

- `lmbda=1.0`, `beta=0.5`
- `temperature=0.6`
- `max_new_tokens=128`
- `min_reward=0.9`
- Learning rate `3e-6`
- Extended schedule (`num_train_epochs=5` or equivalent max steps)

**Telemetry (to be filled in)**

- Training/eval loss trajectory under the stricter filter
- Gradient norms (expected to stay ≤1.0 most of run)
- Baseline Comparison deltas showing tighter token budgets
- Wall-clock / GPU hours relative to fast sweep

Interpretation: this sweep takes longer, but it’s the one we anchor reliability SLAs to. Because the trainer and dataset plumbing are identical, teams can schedule it weekly/biweekly without retooling—just update Hydra overrides.

---

## Comparing the Cadences

Once both runs complete we’ll snapshot the metrics side-by-side. The table will focus on:

| Metric | Gear 1 (Fast) | Gear 2 (High-Quality) |
| --- | --- | --- |
| Train loss (start → end) | _Fill in_ | _Fill in_ |
| Eval loss checkpoints | _Fill in_ | _Fill in_ |
| Success delta (Baseline Comparison) | _Fill in_ | _Fill in_ |
| Token reduction (%) | _Fill in_ | _Fill in_ |
| Wall-clock (hrs) | _Fill in_ | _Fill in_ |

The point isn’t to crown a winner—it’s to show how quickly you can iterate between gears depending on the workflow. Most teams will run Gear 1 after each batch of traces, then schedule Gear 2 for the workflows where reliability or cost targets matter most.

---

## Where RLVR and Self-Play Fit

We still plan to publish a GRPO (RLVR-style) baseline on the exact same GSM8K slice once the trainer stabilizes. RLVR is valuable when you don’t have a trustworthy teacher policy or when you need to explore beyond existing traces. The Atlas SDK already captures the reward adapters and supervision lanes needed for those runs; the difference is that RLVR consumes more compute and requires careful reward shaping. By nailing the two GKD gears first, we get a reliable default loop and can reserve RLVR/self-play for the cases where sparse rewards are the only option.

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

**High-Quality Sweep Command**

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

Both scripts log to Weights & Biases and write metrics under `outputs/gkd_*`. Replace `num-train-epochs` with `max_steps` if you prefer fixed-step schedules.

---

## Closing Thoughts

On-policy distillation gives us dense, per-token guidance without the infrastructure overhead of RLVR. By demonstrating both gears on real SDK traces, we can offer continual learning as a repeatable service: fast cadences for day-to-day improvements, high-quality cadences for SLA workflows, and RLVR/self-play when the problem truly demands exploration. Final metrics coming as soon as the jobs finish.***
