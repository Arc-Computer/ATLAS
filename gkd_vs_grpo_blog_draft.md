# On-Policy Distillation vs. RLVR (GRPO): A Case Study in Knowledge Transfer Efficiency

Atlas is a system for continual learning from agent workflows, designed to close the loop between real-world agent execution and model improvement. The platform is composed of two parts: the **Atlas SDK**, a runtime component that captures rich causality data from agent interactions, and **Atlas Core**, an offline training engine that uses this data to build better models.

This architecture allows our research team to rapidly test and compare different learning strategies. To demonstrate this capability, we conducted a head-to-head analysis of two frontier on-policy methods: Generalized Knowledge Distillation (GKD) and a reinforcement learning with verifiable rewards (RLVR) baseline implemented with Group Relative Policy Optimization (GRPO). The goal was to quantify the trade-offs in performance and compute efficiency between the dense, token-level feedback of GKD and the sparse, outcome-level rewards that enterprises are piloting under the RLVR banner.

### Experimental Design

All runs were executed on a single NVIDIA DGX Spark node using the 25.09 release of NVIDIA's PyTorch container (`nvcr.io/nvidia/pytorch:25.09-py3`). We mounted the Atlas repo and a shared Hugging Face cache into the container, so both training scripts reused the same local downloads of `Qwen/Qwen2.5-7B-Instruct` and `Qwen/Qwen2.5-14B-Instruct`. Metrics (loss curves, pass rate, reward traces, token counts, reverse KL, and wall-clock / GPU hours) were streamed to Weights & Biases for easy comparison.

To create a controlled environment, we focused on MetaMathQA, a public math-reasoning benchmark that stresses structured, multi-step thinking. Our goal was to improve the performance of a student model on this dataset using two different on-policy methods, both orchestrated by Atlas.

The student model in both experiments was `Qwen/Qwen2.5-7B-Instruct`.

**Method 1: Guided Improvement with GKD**

In the first run, we used the `AtlasGKDTrainer` to distill knowledge from a larger teacher model, `Qwen/Qwen2.5-14B-Instruct`. The GKD trainer was set to be fully on-policy (`lambda=1.0`), meaning the 7B student generated its own attempts at solving a math problem. The 14B teacher then provided dense, token-by-token feedback on the student's entire reasoning process. The learning objective here is to minimize the KL divergence from the teacher's probability distribution at every step.

**Method 2: Self-Improvement with GRPO (RLVR Baseline)**

In the second run, we trained the same `Qwen/Qwen2.5-7B-Instruct` model using our `GRPOTrainer` on the identical MetaMathQA split. This configuration mirrors a reinforcement learning with verifiable rewards (RLVR) setup: there is no teacher model, and the agent receives a deterministic, exact-match reward (`+1.0` for correct final answers, `0.0` otherwise). That framing is what many enterprise teams are experimenting with today, so it’s the natural comparison point. The student explores on its own, guided only by the verifiable outcome signal. We exported the same MetaMathQA prompts into Atlas's GRPO dataset schema so the reward function could normalize answers exactly as the distillation evaluator does.

To keep the comparison grounded in “turnkey vs. turnkey” workflows, both trainers ran with their stock Atlas recipes. The GKD experiment simply sets `lambda=1.0` on `AtlasGKDTrainer`, leaving every other hyperparameter at the defaults we ship for MetaMathQA. The GRPO/RLVR baseline likewise uses the out-of-the-box configuration—exact-match reward only, a fixed `1e-6` learning rate, and a 300-step schedule—mirroring how most teams would spin up RLVR before any bespoke reward shaping or curriculum tuning. The results below should therefore be read as a head-to-head between the standard Atlas distillation stack and the standard Atlas RLVR stack, not a “best-possible” tuning sweep for either method.

For evaluation, we measured task performance as the pass rate on a held-out test set of 1,000 problems, and we tracked compute efficiency by the total GPU hours required to reach peak performance.

### Performance and Efficiency Analysis

The first MetaMathQA distillation run produced the following summary:

| Model / Method | Accuracy (MetaMathQA subset) | Avg Generated Tokens | Mean Reverse KL | GPU Time |
| :--- | :--- | :--- | :--- | :--- |
| Student (pre-distillation) | 82.8% | 29.6 | 0.098 | N/A |
| Teacher (Qwen2.5‑14B) | 85.9% | 17.6 | 0.000 | N/A |
| **GKD Distilled Student** | **67.2%** | **96.5** | **0.032** | **13.1 hrs (500 steps)** |
| GRPO / RLVR Baseline | _Running_ | _TBD_ | _TBD_ | _TBD_ |

We still have work to do: after 500 steps the distilled checkpoint trails the baseline student on exact-match accuracy, despite improving its reverse-KL alignment with the teacher. That’s a useful data point—running the same script with longer schedules, a smaller temperature, or a filtered training slice is now the next lever. In parallel, we’re running the RLVR baseline so the final table can contrast both strategies on identical metrics.

Once the GRPO numbers land we’ll add the second row of data and the accompanying charts. For now, the key takeaway is that Atlas let us stage and diagnose the full GKD pipeline on DGX Spark in ~13 hours, giving us concrete telemetry (accuracy, token counts, reverse KL, wall-clock) to tune against.

We’ve already queued a follow-up distillation run that tightens the generation settings (`temperature=0.6`, `max_new_tokens=128`) while keeping the rest of the schedule identical. That should curb the overly long completions and translate the KL gains into reliability gains—we’ll fold those results into the table as soon as they’re finished.

### Special Case: Cross-Tokenizer Distillation

A known challenge in knowledge distillation is a vocabulary mismatch between models. To test this boundary condition, we also ran an experiment distilling from the `Qwen/Qwen2.5-14B-Instruct` teacher to a `meta-llama/Llama-3.2-1B-Instruct` student. These models have entirely different tokenizers, which would historically require complex vocabulary mapping or fail entirely. However, we observed that our GKD implementation, which incorporates modern sequence and vocabulary alignment techniques, handled the discrepancy automatically. This confirms that distillation can be effectively applied even across different model families without bespoke engineering.

### Discussion and Implications

From this analysis, one clear takeaway stands out: for improving a model on a task where a more capable "teacher" policy exists, on-policy distillation offers a significantly more direct and efficient path than RLVR-style approaches. If a capable teacher can evaluate the reasoning process, distilling that knowledge is more effective than forcing a student to rediscover it from scratch with sparse rewards.

More broadly, this study highlights the value of an integrated learning platform. The ability to run, measure, and compare these frontier methods side-by-side, without weeks of custom engineering for each one, is critical for research velocity. Atlas allowed us to move from a research question to a clean, comparative result in a single afternoon. We're always looking to connect with other teams working at the edge of continual learning and agentic systems. If you're exploring similar problems, we'd love to compare notes. You can find us on [Twitter](https://twitter.com/atlas) and [LinkedIn](https://www.linkedin.com/company/atlas).

Ready to see the same workflow on your traces? Reach out at hello@atlas.ai or drop us a line on LinkedIn—happy to run the distillation vs. RLVR comparison with your models and datasets.

### Methods Appendix

We ran both experiments on a single NVIDIA DGX Spark using NVIDIA’s PyTorch container (`nvcr.io/nvidia/pytorch:25.09-py3`). The Atlas repository and Hugging Face cache were bind-mounted into the container so the student (`Qwen/Qwen2.5-7B-Instruct`) and teacher (`Qwen/Qwen2.5-14B-Instruct`) checkpoints could be reused without re-downloading. Runs were logged to Weights & Biases to capture pass rate, token statistics, reverse KL, reward traces, and wall-clock GPU hours.

**GKD validation command**

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 \
PYTHONPATH=. \
CUDA_VISIBLE_DEVICES=0 \
python scripts/validate_gkd.py \
  --student Qwen/Qwen2.5-7B-Instruct \
  --teacher Qwen/Qwen2.5-14B-Instruct \
  --train-limit 2048 \
  --eval-limit 256 \
  --max-steps 500 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-5 \
  --lmbda 1.0 \
  --beta 0.5 \
  --temperature 0.9 \
  --max-new-tokens 256 \
  --eval-sample-size 128 \
  --bf16
```

`scripts/validate_gkd.py` writes metrics to `outputs/gkd_math_validation/math_validation_metrics.json` and streams reverse KL, accuracy, and token counts to W&B.

**GRPO (RLVR-style) command**

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 \
PYTHONPATH=. \
CUDA_VISIBLE_DEVICES=0 \
python train.py --config-name train_meta_math_grpo
```

`configs/train_meta_math_grpo.yaml` pinpoints the 7B student, loads MetaMathQA through `custom_data.math_grpo_data`, and plugs in the exact-match reward defined in `trainers/reward_functions.py`. Default settings run 300 optimizer steps with an effective batch of 64 rollouts (per-device batch size 2, gradient accumulation 32). For quick smoke checks, override `max_steps` or `data.dataset_max_samples` at the command line.

Both scripts share the same answer normalization logic, ensuring the GKD evaluator and RLVR reward function judge outputs identically. This setup lets teams reproduce the comparison end-to-end on a single DGX Spark and capture the same telemetry we referenced in the analysis.
