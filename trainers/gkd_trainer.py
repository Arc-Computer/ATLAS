"""GKD trainer wrapper for Atlas runtime trace distillation.

This module provides AtlasGKDTrainer, a minimal wrapper around TRL's GKDTrainer
that integrates with Atlas' Postgres-backed dataset infrastructure and adds
baseline comparison metrics tracking for evaluating distillation quality.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

try:
    from trl import GKDTrainer, GKDConfig
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "trl>=0.12.0 is required for GKD training. "
        "Install with: pip install 'trl>=0.12.0'"
    ) from exc

from trainers.gkd_dataset import build_gkd_dataset
from trainers.gkd_evaluator import BaselineMetricsCallback

logger = logging.getLogger(__name__)


class AtlasGKDTrainer(GKDTrainer):
    """
    Atlas-specific wrapper around TRL's GKDTrainer for on-policy distillation.

    This trainer extends TRL's GKDTrainer with:
    - Direct Postgres dataset loading via build_gkd_dataset()
    - baseline comparison metrics callback for tracking distillation quality
    - Configurable teacher model loading (local, API, or Atlas teacher)

    The trainer implements on-policy reverse-KL distillation, which is 9-30×
    more compute-efficient than GRPO while preserving learning quality.

    Args:
        model: Student model to be trained (smaller, faster model)
        teacher_model: Teacher model to distill from (larger, more capable model)
        args: GKDConfig with training hyperparameters
        db_url: PostgreSQL connection string for loading Atlas traces
        min_reward: Minimum reward threshold for filtering traces (default: 0.8)
        learning_key: Optional task-specific filter (e.g., "crm_workflows")
        baseline_success: Baseline success rate for calculating baseline comparison delta
        baseline_tokens: Baseline token count for calculating efficiency gains
        processing_class: Tokenizer or processor for the student model
        **kwargs: Additional arguments passed to GKDTrainer

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from trl import GKDConfig
        >>>
        >>> student = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B")
        >>> teacher = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-14B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
        >>>
        >>> args = GKDConfig(
        ...     output_dir="outputs/gkd",
        ...     per_device_train_batch_size=4,
        ...     lmbda=1.0,  # Fully on-policy
        ...     beta=0.5,
        ... )
        >>>
        >>> trainer = AtlasGKDTrainer(
        ...     model=student,
        ...     teacher_model=teacher,
        ...     args=args,
        ...     db_url="postgresql://localhost:5432/atlas",
        ...     min_reward=0.8,
        ...     processing_class=tokenizer,
        ... )
        >>>
        >>> trainer.train()
        >>> trainer.save_model("outputs/gkd/final")
    """

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        teacher_model: Optional[PreTrainedModel] = None,
        args: Optional[GKDConfig] = None,
        db_url: Optional[str] = None,
        min_reward: float = 0.8,
        learning_key: Optional[str] = None,
        baseline_success: float = 0.0,
        baseline_tokens: Optional[float] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        **kwargs,
    ):
        """Initialize AtlasGKDTrainer with Postgres dataset loading."""

        # Load conversations from Postgres
        if db_url is None:
            raise ValueError(
                "db_url is required for AtlasGKDTrainer. "
                "Set db_url or ATLAS_DB_URL environment variable."
            )

        logger.info(
            "Loading GKD dataset from Postgres (min_reward=%.2f, learning_key=%s)",
            min_reward,
            learning_key,
        )

        train_dataset, eval_dataset = build_gkd_dataset(
            db_url=db_url,
            min_reward=min_reward,
            learning_key=learning_key,
            eval_split=args.eval_split if hasattr(args, "eval_split") else 0.1,
            seed=args.seed if hasattr(args, "seed") else 42,
        )

        logger.info(
            "Loaded datasets: train=%d, eval=%d conversations",
            len(train_dataset),
            len(eval_dataset),
        )

        # Add baseline comparison evaluation callback if baseline metrics provided
        if baseline_success > 0 or baseline_tokens is not None:
            callbacks = kwargs.get("callbacks", [])
            stage2_callback = BaselineMetricsCallback(
                baseline_success=baseline_success,
                baseline_tokens=baseline_tokens,
            )
            callbacks.append(stage2_callback)
            kwargs["callbacks"] = callbacks
            logger.info(
                "Added baseline comparison metrics tracking (baseline_success=%.2f, baseline_tokens=%s)",
                baseline_success,
                baseline_tokens,
            )

        # Initialize TRL's GKDTrainer
        # This handles all the distillation logic:
        # - On-policy student generation (controlled by lmbda)
        # - Teacher feedback on student outputs
        # - Generalized JSD loss (controlled by beta)
        super().__init__(
            model=model,
            teacher_model=teacher_model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            **kwargs,
        )

        logger.info(
            "AtlasGKDTrainer initialized with lmbda=%.1f (on-policy fraction), beta=%.1f (KL balance)",
            args.lmbda if hasattr(args, "lmbda") else 1.0,
            args.beta if hasattr(args, "beta") else 0.5,
        )


def create_gkd_trainer_from_config(
    student_model_path: str,
    teacher_model_path: str,
    db_url: str,
    args: GKDConfig,
    **kwargs,
) -> AtlasGKDTrainer:
    """
    Factory function to create AtlasGKDTrainer from model paths and config.

    This is a convenience function for Hydra instantiation or programmatic
    trainer creation without manually loading models.

    Args:
        student_model_path: HuggingFace model ID or path for student
        teacher_model_path: HuggingFace model ID or path for teacher
        db_url: PostgreSQL connection string
        args: GKDConfig with training parameters
        **kwargs: Additional arguments passed to AtlasGKDTrainer

    Returns:
        Initialized AtlasGKDTrainer ready for training

    Example:
        >>> from trl import GKDConfig
        >>>
        >>> args = GKDConfig(
        ...     output_dir="outputs/gkd",
        ...     lmbda=1.0,
        ...     beta=0.5,
        ... )
        >>>
        >>> trainer = create_gkd_trainer_from_config(
        ...     student_model_path="Qwen/Qwen2-7B-Instruct",
        ...     teacher_model_path="Qwen/Qwen2-14B-Instruct",
        ...     db_url="postgresql://localhost:5432/atlas",
        ...     args=args,
        ...     min_reward=0.8,
        ... )
        >>>
        >>> trainer.train()
    """
    logger.info("Loading student model from: %s", student_model_path)
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_path,
        torch_dtype="auto",
        device_map="auto",
    )

    logger.info("Loading teacher model from: %s", teacher_model_path)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_path,
        torch_dtype="auto",
        device_map="auto",
    )

    logger.info("Loading tokenizer from: %s", student_model_path)
    tokenizer = AutoTokenizer.from_pretrained(student_model_path)

    # Ensure padding token is set (required for batch processing)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token: %s", tokenizer.eos_token)

    return AtlasGKDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=args,
        db_url=db_url,
        processing_class=tokenizer,
        **kwargs,
    )
