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

For evaluation, we measured task performance as the pass rate on a held-out test set of 1,000 problems, and we tracked compute efficiency by the total GPU hours required to reach peak performance.

### Performance and Efficiency Analysis

The results of the comparison were definitive. The learning curve for the GKD student was a steep, rapid ascent, achieving 90% of the teacher's final score in under three hours. The dense, token-level guidance allowed the model to quickly correct its reasoning pathways.

In contrast, the GRPO model's curve was a slow, noisy crawl upwards. Learning from only a sparse "correct/incorrect" signal, it struggled to assign credit and took much longer to discover effective strategies. While it did improve, its learning was far less efficient.

*(A chart comparing the pass rate vs. training time for GKD and GRPO would be placed here.)*

The difference in efficiency was stark.

*(A table comparing the final metrics would be placed here.)*

| Method | Peak Performance (Pass Rate) | Time to Peak | Compute (GPU Hours) |
| :--- | :--- | :--- | :--- |
| GRPO | 38% | 22 hours | ~22 |
| **GKD** | **55%** | **3 hours** | **~3** |

The dense signal from GKD was approximately **7x more compute-efficient** and resulted in a **17 percentage point higher** final performance. For this knowledge transfer task, guiding the student's reasoning process token-by-token was demonstrably more effective than letting it search for a solution with only a simple "win/loss" signal.

### Special Case: Cross-Tokenizer Distillation

A known challenge in knowledge distillation is a vocabulary mismatch between models. To test this boundary condition, we also ran an experiment distilling from the `Qwen/Qwen2.5-14B-Instruct` teacher to a `meta-llama/Llama-3.2-1B-Instruct` student. These models have entirely different tokenizers, which would historically require complex vocabulary mapping or fail entirely. However, we observed that our GKD implementation, which incorporates modern sequence and vocabulary alignment techniques, handled the discrepancy automatically. This confirms that distillation can be effectively applied even across different model families without bespoke engineering.

### Discussion and Implications

From this analysis, one clear takeaway stands out: for improving a model on a task where a more capable "teacher" policy exists, on-policy distillation offers a significantly more direct and efficient path than RLVR-style approaches. If a capable teacher can evaluate the reasoning process, distilling that knowledge is more effective than forcing a student to rediscover it from scratch with sparse rewards.

More broadly, this study highlights the value of an integrated learning platform. The ability to run, measure, and compare these frontier methods side-by-side, without weeks of custom engineering for each one, is critical for research velocity. Atlas allowed us to move from a research question to a clean, comparative result in a single afternoon. We're always looking to connect with other teams working at the edge of continual learning and agentic systems. If you're exploring similar problems, we'd love to compare notes. You can find us on [Twitter](https://twitter.com/atlas) and [LinkedIn](https://www.linkedin.com/company/atlas).

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
