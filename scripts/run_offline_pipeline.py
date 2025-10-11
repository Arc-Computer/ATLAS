#!/usr/bin/env python3
"""
Helper CLI that stitches together the offline GRPO workflow for Atlas Core.

Given a runtime trace export (typically produced by the atlas-sdk CLI), this
script launches `train.py` with the correct Hydra overrides so teams can run
`export → train` without hand assembling commands. The SDK can invoke this
script directly as part of a one-touch workflow.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Atlas GRPO training on exported runtime traces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--export-path",
        required=True,
        type=Path,
        help="Path to the runtime traces JSONL file exported from the SDK.",
    )
    parser.add_argument(
        "--config-name",
        default="train",
        help="Hydra config to load (e.g., `train` or `examples/quickstart`).",
    )
    parser.add_argument(
        "--data-config",
        default="runtime_traces",
        help="Hydra data config override (maps to configs/data/<name>.yaml).",
    )
    parser.add_argument(
        "--trainer-config",
        default="grpo",
        help="Hydra trainer config override (maps to configs/trainer/<name>.yaml).",
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="Optional Hydra model config override (configs/model/<name>.yaml).",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=None,
        help="Override `data.eval_split_ratio`.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override `data.dataset_max_samples`.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override `output_dir` for the training run.",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Override `wandb_project`.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Override `wandb_run_name`.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override (repeatable), e.g. `trainer.max_steps=500`.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without executing it.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    export_path = args.export_path.expanduser()
    if not export_path.exists():
        raise FileNotFoundError(
            f"Export file not found: {export_path}. "
            "Run the atlas-sdk export command before invoking this helper."
        )

    overrides: list[str] = [
        f"data@_global_={args.data_config}",
        f"trainer@_global_={args.trainer_config}",
        f"data.dataset_path={export_path}",
    ]

    if args.model_config:
        overrides.append(f"model@_global_={args.model_config}")
    if args.eval_ratio is not None:
        overrides.append(f"data.eval_split_ratio={args.eval_ratio}")
    if args.max_samples is not None:
        overrides.append(f"data.dataset_max_samples={args.max_samples}")
    if args.output_dir:
        overrides.append(f"output_dir={args.output_dir}")
    if args.wandb_project:
        overrides.append(f"wandb_project={args.wandb_project}")
    if args.wandb_run_name:
        overrides.append(f"wandb_run_name={args.wandb_run_name}")

    overrides.extend(args.override)

    cmd = [sys.executable, "train.py", "--config-name", args.config_name, *overrides]

    print("Running:", " ".join(shlex.quote(part) for part in cmd))
    if args.dry_run:
        return 0

    completed = subprocess.run(cmd, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
