"""Baseline evaluation metrics tracking for GKD training.

This module provides callbacks and utilities for monitoring GKD training progress
against baseline performance:
- Success delta: Improvement over baseline success rate
- Token reduction: Reduction in tokens per episode vs baseline
- Learning metrics: Cue hit, adoption, reward/token deltas

These metrics provide continuous monitoring during training to validate distillation quality.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class BaselineMetricsCallback(TrainerCallback):
    """
    Callback to track baseline comparison metrics during GKD training.

    This callback augments standard evaluation metrics with baseline comparison
    measurements that track distillation quality and learning preservation:

    **Success Metrics**:
    - metrics/success_delta: Current success rate - baseline
    - metrics/meets_target: Boolean flag for whether improvement achieved

    **Efficiency Metrics**:
    - metrics/token_reduction_pct: Percentage reduction in tokens vs baseline
    - metrics/token_target_met: Boolean flag for whether reduction achieved

    **Learning Metrics** (if available in eval outputs):
    - metrics/cue_hit_rate: Teacher intervention appropriateness
    - metrics/adoption_rate: Student following teacher guidance
    - metrics/reward_delta: Improvement from teacher interventions

    These metrics are logged to WandB/TensorBoard alongside standard training
    metrics and can be used for early stopping or hyperparameter selection.

    Args:
        baseline_success: Baseline task success rate (0.0-1.0) for calculating delta
        baseline_tokens: Baseline average tokens per episode for calculating efficiency

    Example:
        >>> from trl import GKDConfig
        >>> from trainers.gkd_trainer import AtlasGKDTrainer
        >>> from trainers.gkd_evaluator import BaselineMetricsCallback
        >>>
        >>> callback = BaselineMetricsCallback(
        ...     baseline_success=0.75,  # 75% baseline success rate
        ...     baseline_tokens=1200,   # 1200 tokens average
        ... )
        >>>
        >>> args = GKDConfig(output_dir="outputs/gkd")
        >>> trainer = AtlasGKDTrainer(
        ...     model=student,
        ...     teacher_model=teacher,
        ...     args=args,
        ...     db_url="...",
        ...     callbacks=[callback],
        ... )
        >>>
        >>> trainer.train()
        # During training, metrics logged:
        # - metrics/success_delta: 0.12  (87% - 75%)
        # - metrics/meets_target: True
        # - metrics/token_reduction_pct: 35.0
        # - metrics/token_target_met: True
    """

    def __init__(
        self,
        baseline_success: float = 0.0,
        baseline_tokens: Optional[float] = None,
    ):
        """
        Initialize baseline comparison metrics callback.

        Args:
            baseline_success: Baseline success rate (0.0-1.0). Set to 0.0 if unknown.
            baseline_tokens: Baseline average tokens per episode. Set to None if unknown.
        """
        self.baseline_success = baseline_success
        self.baseline_tokens = baseline_tokens

        logger.info(
            "BaselineMetricsCallback initialized: baseline_success=%.2f, baseline_tokens=%s",
            baseline_success,
            baseline_tokens if baseline_tokens is not None else "N/A",
        )

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float],
        **kwargs,
    ) -> None:
        """
        Augment evaluation metrics with baseline comparison measurements.

        This method is called after each evaluation step and adds baseline comparison-specific
        metrics to the metrics dict, which are then logged to WandB/TensorBoard.

        Args:
            args: Training arguments
            state: Current trainer state
            control: Trainer control flow
            metrics: Evaluation metrics dict (modified in-place)
            **kwargs: Additional callback arguments
        """

        # === Success Delta Metrics ===
        # Calculate improvement over baseline success rate
        current_success = metrics.get("eval_success_rate", 0.0)
        if current_success > 0 or self.baseline_success > 0:
            success_delta = current_success - self.baseline_success
            metrics["metrics/success_delta"] = success_delta
            metrics["metrics/meets_target"] = success_delta >= 0.10  # ✅ ≥10 pp

            logger.debug(
                "baseline comparison success: current=%.2f%%, baseline=%.2f%%, delta=%.2f pp",
                current_success * 100,
                self.baseline_success * 100,
                success_delta * 100,
            )

        # === Token Efficiency Metrics ===
        # Calculate token reduction percentage
        current_tokens = metrics.get("eval_avg_tokens", 0.0)
        if current_tokens > 0 and self.baseline_tokens is not None:
            token_reduction = (self.baseline_tokens - current_tokens) / self.baseline_tokens
            metrics["metrics/token_reduction_pct"] = token_reduction * 100
            metrics["metrics/token_target_met"] = token_reduction >= 0.30  # ✅ ≥30%

            logger.debug(
                "baseline comparison efficiency: current=%.0f tokens, baseline=%.0f tokens, reduction=%.1f%%",
                current_tokens,
                self.baseline_tokens,
                token_reduction * 100,
            )

        # === Learning Metrics ===
        # Pass through Atlas learning metrics if present in eval outputs
        # These align with definitions in how-we-define-learning.md

        # Cue hit rate: Teacher intervention triggered appropriately
        if "eval_cue_hit_rate" in metrics:
            metrics["metrics/cue_hit_rate"] = metrics["eval_cue_hit_rate"]

        # Adoption rate: Student follows teacher guidance
        if "eval_adoption_rate" in metrics:
            metrics["metrics/adoption_rate"] = metrics["eval_adoption_rate"]

        # Reward delta: Improvement from teaching
        if "eval_reward_delta" in metrics:
            metrics["metrics/reward_delta"] = metrics["eval_reward_delta"]

        # Token delta: Token efficiency from teaching
        if "eval_token_delta" in metrics:
            metrics["metrics/token_delta"] = metrics["eval_token_delta"]

        # Transfer success: Learnings applied to new conversations
        if "eval_transfer_success" in metrics:
            metrics["metrics/transfer_success"] = metrics["eval_transfer_success"]

        # === Summary Logging ===
        # Log high-level baseline comparison status
        meets_success_target = metrics.get("metrics/meets_target", False)
        meets_token_target = metrics.get("metrics/token_target_met", False)

        if meets_success_target and meets_token_target:
            logger.info(
                "✅ baseline comparison targets MET: success delta=%.1f pp, token reduction=%.1f%%",
                metrics.get("metrics/success_delta", 0.0) * 100,
                metrics.get("metrics/token_reduction_pct", 0.0),
            )
        elif meets_success_target:
            logger.info(
                "⚠️  baseline comparison partial: success target MET (%.1f pp), token target NOT MET (%.1f%%)",
                metrics.get("metrics/success_delta", 0.0) * 100,
                metrics.get("metrics/token_reduction_pct", 0.0),
            )
        elif meets_token_target:
            logger.info(
                "⚠️  baseline comparison partial: token target MET (%.1f%%), success target NOT MET (%.1f pp)",
                metrics.get("metrics/token_reduction_pct", 0.0),
                metrics.get("metrics/success_delta", 0.0) * 100,
            )
        else:
            logger.info(
                "❌ baseline comparison targets NOT MET: success delta=%.1f pp (need ≥10), token reduction=%.1f%% (need ≥30)",
                metrics.get("metrics/success_delta", 0.0) * 100,
                metrics.get("metrics/token_reduction_pct", 0.0),
            )

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """
        Log baseline metrics at training start.

        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
            **kwargs: Additional callback arguments
        """
        logger.info(
            "Starting GKD training with baseline comparison metrics: success=%.2f%%, tokens=%s",
            self.baseline_success * 100,
            f"{self.baseline_tokens:.0f}" if self.baseline_tokens is not None else "N/A",
        )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """
        Log final baseline comparison status at training end.

        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
            **kwargs: Additional callback arguments
        """
        # Retrieve final eval metrics from state
        if state.log_history:
            # Find most recent eval metrics
            final_metrics = {}
            for log_entry in reversed(state.log_history):
                if "eval_loss" in log_entry:
                    final_metrics = log_entry
                    break

            if final_metrics:
                success_delta = final_metrics.get("metrics/success_delta")
                token_reduction = final_metrics.get("metrics/token_reduction_pct")

                logger.info(
                    "Training completed. Final baseline comparison metrics: "
                    "success_delta=%s, token_reduction=%s",
                    f"{success_delta*100:.1f} pp" if success_delta is not None else "N/A",
                    f"{token_reduction:.1f}%" if token_reduction is not None else "N/A",
                )
        else:
            logger.info("Training completed. No evaluation metrics available.")


def compute_baseline_summary(
    eval_results: Dict[str, float],
    baseline_success: float,
    baseline_tokens: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute baseline comparison summary statistics from evaluation results.

    This is a utility function for standalone evaluation (outside of training)
    to compute baseline comparison metrics from a dict of evaluation results.

    Args:
        eval_results: Dict of evaluation metrics (e.g., from trainer.evaluate())
        baseline_success: Baseline success rate (0.0-1.0)
        baseline_tokens: Baseline average tokens per episode

    Returns:
        Dict with baseline comparison summary:
        - success_delta: Absolute improvement (pp)
        - success_relative: Relative improvement (%)
        - token_reduction_pct: Token reduction percentage
        - meets_success_target: Boolean
        - meets_token_target: Boolean
        - meets_all_targets: Boolean

    Example:
        >>> eval_results = trainer.evaluate()
        >>> summary = compute_baseline_summary(
        ...     eval_results,
        ...     baseline_success=0.75,
        ...     baseline_tokens=1200,
        ... )
        >>> print(f"Success delta: {summary['success_delta']*100:.1f} pp")
        Success delta: 12.5 pp
        >>> print(f"Meets targets: {summary['meets_all_targets']}")
        Meets targets: True
    """
    current_success = eval_results.get("eval_success_rate", 0.0)
    success_delta = current_success - baseline_success
    success_relative = success_delta / baseline_success if baseline_success > 0 else 0.0

    summary = {
        "success_delta": success_delta,
        "success_relative": success_relative,
        "meets_success_target": success_delta >= 0.10,
    }

    if baseline_tokens is not None:
        current_tokens = eval_results.get("eval_avg_tokens", 0.0)
        token_reduction = (baseline_tokens - current_tokens) / baseline_tokens
        summary["token_reduction_pct"] = token_reduction * 100
        summary["meets_token_target"] = token_reduction >= 0.30
        summary["meets_all_targets"] = (
            summary["meets_success_target"] and summary["meets_token_target"]
        )
    else:
        summary["token_reduction_pct"] = None
        summary["meets_token_target"] = None
        summary["meets_all_targets"] = summary["meets_success_target"]

    return summary
