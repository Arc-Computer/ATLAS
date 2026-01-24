from __future__ import annotations

import subprocess
import sys


def _exec_module(module: str, args: list[str]) -> int:
    cmd = [sys.executable, "-m", module, *args]
    return subprocess.call(cmd)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(
            "atlas-core <command> [args]\n\n"
            "Commands:\n"
            "  train             Launch Hydra-driven training runs\n"
            "  offline-pipeline   Run the offline GRPO pipeline helper\n"
        )
        raise SystemExit(0)

    command = sys.argv[1]
    args = sys.argv[2:]

    if command == "train":
        raise SystemExit(_exec_module("atlas_core.cli.train", args))
    if command in {"offline-pipeline", "offline_pipeline"}:
        raise SystemExit(_exec_module("atlas_core.cli.offline_pipeline", args))

    print(f"Unknown command: {command}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
