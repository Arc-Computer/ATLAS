import logging
import os
import random
import re
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from transformers.trainer_utils import get_last_checkpoint
from typing import Any, Dict, Optional, cast

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def wandb_init(cfg, run_name: str, group_name: str, log_dir: str):
    import wandb
    from omegaconf import OmegaConf

    config_dict = cast(
        Dict[str, Any],
        OmegaConf.to_container(
        cfg,
        resolve=True,
        throw_on_missing=False,
    ))
    config_dict["log_dir"] = log_dir
    config_dict["wandb_run_name"] = run_name
    config_dict["wandb_group_name"] = group_name

    wandb_run = wandb.init(
        project=cfg.wandb_project,
        group=group_name[:127],
        name=run_name[:127],
        config=config_dict,
    )
    return wandb


def get_checkpoint(output_dir):
    if os.path.isdir(output_dir):
        return get_last_checkpoint(output_dir)
    return None


def get_total_devices():
    world_size = os.environ.get("WORLD_SIZE")
    if world_size is not None:
        return int(world_size)
    return 1


def compute_accumulation_steps(train_batch_size, per_device_train_batch_size):
    total_devices = get_total_devices()

    div = per_device_train_batch_size*total_devices
    steps = train_batch_size/div
    if not steps.is_integer():
        raise ValueError(
            "train_batch_size must be divisible by "
            f"per_device_batch*total_devices={div}"
        )
    return int(steps)


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    if "LOCAL_RANK" in os.environ:
        is_main_process = int(os.environ["LOCAL_RANK"]) == 0
    elif "RANK" in os.environ:
        is_main_process = int(os.environ["RANK"]) == 0
    else:
        is_main_process = True

    def _resolve_grad_accum(target_cfg: DictConfig) -> int | None:
        if target_cfg is None or not isinstance(target_cfg, DictConfig):
            return None
        if not OmegaConf.is_missing(target_cfg, "gradient_accumulation_steps"):
            return target_cfg.gradient_accumulation_steps

        args_cfg = target_cfg.get("args") if isinstance(target_cfg.get("args"), DictConfig) else None
        if args_cfg is not None and not OmegaConf.is_missing(args_cfg, "gradient_accumulation_steps"):
            return args_cfg.gradient_accumulation_steps

        train_batch = None
        if not OmegaConf.is_missing(target_cfg, "train_batch_size"):
            train_batch = target_cfg.train_batch_size
        elif args_cfg is not None and not OmegaConf.is_missing(args_cfg, "train_batch_size"):
            train_batch = args_cfg.train_batch_size
        elif not OmegaConf.is_missing(cfg, "train_batch_size"):
            train_batch = cfg.train_batch_size

        per_device_batch = None
        if not OmegaConf.is_missing(target_cfg, "per_device_train_batch_size"):
            per_device_batch = target_cfg.per_device_train_batch_size
        elif args_cfg is not None and not OmegaConf.is_missing(args_cfg, "per_device_train_batch_size"):
            per_device_batch = args_cfg.per_device_train_batch_size
        elif not OmegaConf.is_missing(cfg, "per_device_train_batch_size"):
            per_device_batch = cfg.per_device_train_batch_size

        if train_batch is None or per_device_batch is None:
            return None

        return compute_accumulation_steps(
            train_batch_size=train_batch,
            per_device_train_batch_size=per_device_batch,
        )

    accumulation_steps = _resolve_grad_accum(cfg)
    trainer_cfg = cfg.get("trainer", None)
    trainer_accum = _resolve_grad_accum(trainer_cfg)

    resolved_value: Optional[int] = None
    if trainer_accum is not None:
        trainer_cfg.gradient_accumulation_steps = trainer_accum
        resolved_value = trainer_accum

    if accumulation_steps is not None:
        cfg.gradient_accumulation_steps = accumulation_steps
        resolved_value = accumulation_steps
    elif resolved_value is not None and OmegaConf.is_missing(cfg, "gradient_accumulation_steps"):
        cfg.gradient_accumulation_steps = resolved_value

    if resolved_value is None:
        logger.warning("gradient_accumulation_steps could not be inferred; "
                       "set it explicitly in your config.")
    else:
        logger.info(f"Accumulation steps {resolved_value} ----")

    using_wandb = False
    if isinstance(cfg.report_to, str):
        using_wandb = cfg.report_to == 'wandb'
    elif cfg.report_to is not None:
        for v in cfg.report_to:
            using_wandb = using_wandb or (v == 'wandb')

    if using_wandb and is_main_process:
        wandb = wandb_init(
            cfg=cfg,
            group_name=cfg.wandb_group_name,
            run_name=cfg.wandb_run_name,
            log_dir=cfg.output_dir,
        )

    tokenizer = hydra.utils.instantiate(cfg.make_tokenizer_fn)

    datasets = hydra.utils.instantiate(
        cfg.make_dataset_fn, tokenizer=tokenizer)

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        **datasets,
    )

    print('Model initialized!!!')

    last_checkpoint = get_checkpoint(cfg.output_dir)
    if not last_checkpoint and cfg.resume_from is not None:
        last_checkpoint = get_checkpoint(cfg.resume_from)
    if last_checkpoint:
        logger.info("Found checkpoint, resuming training run from "
                    f"{last_checkpoint}.")
    else:
        logger.info("No existing checkpoint, initializing new model")

    logger.info(f"Training  {datetime.now()}")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    logger.info(f"Training complete {datetime.now()}")

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    if cfg.save_final_model:
        logger.info(f"Saving final model at {cfg.output_dir}")
        trainer.model.config.use_cache = True
        trainer.save_model(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        logger.info(f"Done saving {datetime.now()}")

    if is_main_process and cfg.push_to_hub:
        tags = cfg.tags if cfg.tags is not None else []
        trainer.create_model_card({"tags": tags})
    if cfg.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    if is_main_process and cfg.call_post_training is not None:

        hydra.utils.instantiate(cfg.call_post_training)


if __name__ == "__main__":
    main()
