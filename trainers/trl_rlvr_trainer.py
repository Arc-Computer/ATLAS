from __future__ import annotations

from typing import Any, Optional

from datasets import Dataset
from transformers import PreTrainedTokenizerBase


class TrlRLVRTrainer:
    """Thin wrapper around TRL's GRPOTrainer for RLVR baselines."""

    def __init__(
        self,
        model: Any,
        args: Any,
        reward_funcs: Any,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        peft_config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        from trl import GRPOTrainer

        processing_class = processing_class or tokenizer
        self._trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            peft_config=peft_config,
            **kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._trainer, name)
