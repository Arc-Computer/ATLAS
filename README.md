# ATLAS: A Continual Learning Architecture for Production AI

<div align="center">

<img src="public/ATLAS.png" alt="ATLAS Hero" width="900" style="border-radius: 12px;">
<br>
[![ATLAS-8B-Thinking](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ATLAS--8B--Thinking-blue)](https://huggingface.co/Arc-Intelligence/ATLAS-8B-Thinking)
[![ATLAS-8B-Instruct](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ATLAS--8B--Instruct-blue)](https://huggingface.co/Arc-Intelligence/ATLAS-8B-Instruct)
[![Arc-ATLAS-Teach Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Arc--ATLAS--Teach-green)](https://huggingface.co/datasets/Arc-Intelligence/Arc-ATLAS-Teach-v0)

</div>

ATLAS is an architecture for production teams that need AI agents to improve from user interactions and feedback *after* deployment. It wraps any existing agent framework with the components required to create a closed-loop, continual learning system:

1.  **Reasoning Core**: A Teacher-Student model pair that enhances agent capabilities.
2.  **Reward System (RIM)**: Turns implicit and explicit user feedback (edits, approvals, tool usage) into a dense reward signal.
3.  **Learning Engine**: Uses online (GEPA) and offline (GRPO) methods to update models based on rewards.
4.  **Persistent Memory**: Stores all interactions for analysis and retraining.

Together, they form the complete learning loop shown below.

Teams have deployed this loop in dual-control environments and high-stakes operations, pairing ATLAS with human supervisors to ship reliable agents in production ([Introducing ATLAS](https://www.arc.computer/blog/introducing-atlas), [Navigating Dual-Control Environments](https://www.arc.computer/blog/navigating-dual-control-environments)). The system captures interaction data, scores it for quality, adapts the models, and redeploys the improved version.

<div align="center">
<img src="public/system-architecture.png" alt="ATLAS System Architecture Diagram" width="800" style="border-radius: 12px;">
<br>
<em>Figure: ATLAS keeps the agent in a learn–evaluate–update cycle.</em>
</div>

📄 **[Read the ATLAS Technical Report](https://docs.arc.computer/technical-report)** for comprehensive methodology and performance analysis.

📚 **[Full Documentation](https://docs.arc.computer)** for complete guides, API reference, and examples.

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
- **Expected Outcome**: A set of optimized teaching prompts in `optimized_prompts.json`. This process takes ~2 hours and costs ~$10 in API fees, delivering a performance improvement of up to +165% (measured as reward delta on evaluation tasks). See [Supercharging RL with Online Optimization](https://www.arc.computer/blog/supercharging-rl-with-online-optimization) for experimental setup.

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
  ```
- **Expected Outcome**: A JSON object containing a reward score (0.0-1.0) and a detailed rationale. The system achieves 93.7% accuracy on RewardBench V2, outperforming all public models. Benchmarks and judge configuration are detailed in [ATLAS Reward System](https://www.arc.computer/blog/atlas-reward-system).

### Path 3 — Build a Custom Teacher Model

Use the full **Learning Engine (GRPO)** to train a new **Reasoning Core** from scratch on your own data. This is for advanced use cases requiring deep domain specialization. Follow the [Custom Teacher Training Guide](https://docs.arc.computer/first-experiment).

- **Prerequisites**: Multi-GPU environment (4-8x A100/H100 recommended).
- **Commands (Minimal Smoke Test)**:
  ```bash
  # Phase 1: SFT Warmup (1 epoch)
  scripts/launch.sh 1 configs/run/teacher_sft.yaml report_to=null save_final_model=false num_train_epochs=1

  # Phase 2: RL Training with vLLM (4 steps)
  scripts/launch_with_server.sh 1 1 configs/run/teacher_rcl.yaml report_to=null max_steps=4 eval_steps=1
  ```
- **Expected Outcome**: A custom-trained teacher model checkpoint. A full training run can achieve a +15.7% average accuracy lift on student agents.

---

## Core Distinctions

<details>
<summary><strong>Why not just fine-tune?</strong></summary>

Fine-tuning (or RLHF) creates a static, updated version of a model. It is compute-intensive and risks catastrophic forgetting. When the world changes, you must repeat the entire process.

ATLAS creates a **dynamic, continual learning loop**. The teacher-student architecture separates foundational knowledge from task-specific adaptation. This means:
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

Conda is recommended for environment management. The repository has been validated with Python 3.11 and 3.12.

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