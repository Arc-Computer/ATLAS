try:
    from .grpo import GRPOTrainer
    from .grpo_config import GRPOConfig
    from .teacher_trainers import TeacherGRPOTrainer
    from .gkd_trainer import AtlasGKDTrainer
    from .trl_rlvr_trainer import TrlRLVRTrainer
    from .data_reward_scorer import (
        DataScorerArgs,
        DataTeacherRewardScorer,
        DataConcatenatorArgs,
        DataCompletionConcatenator,
    )
    from RIM.reward_adapter import RIMReward
    from .runtime_dataset import load_runtime_traces, flatten_traces_for_training
except (ImportError, RuntimeError):
    # Optional dependencies (e.g. apex) may not be installed in lightweight setups.
    pass
