# ArcOps-Cyber Offline Workflow (GRPO Day 0)

## 1. Export traces from Postgres

```bash
# Student-only baseline runs
python -m atlas.cli.main export \
  --database-url postgresql://atlas:atlas@localhost:5433/atlas \
  --status succeeded \
  --output paper_assets/arcops_cyber/traces/student_baseline.jsonl \
  --limit 200

# Guided Teacher runs
python -m atlas.cli.main export \
  --database-url postgresql://atlas:atlas@localhost:5433/atlas \
  --status succeeded \
  --output paper_assets/arcops_cyber/traces/teacher_guided.jsonl \
  --limit 200 \
  --session-id <ids if filtering>
```

Adjust `--limit`/`--session-id` once the full batch runs are complete.

## 2. Launch GRPO training via the new `atlas train` CLI

```bash
python -m atlas.cli.main train \
  --database-url postgresql://atlas:atlas@localhost:5433/atlas \
  --status succeeded \
  --output paper_assets/arcops_cyber/traces/teacher_guided.jsonl \
  --atlas-core-path /path/to/atlas_core \
  --config-name run/teacher_grpo \
  --data-config configs/data/runtime_traces.yaml \
  --override data.dataset_path=paper_assets/arcops_cyber/traces/teacher_guided.jsonl \
  --override data.eval_split_ratio=0.1 \
  --override trainer.total_steps=1500 \
  --override trainer.eval_interval=100 \
  --output-dir paper_assets/arcops_cyber/grpo_runs/teacher_v0 \
  --wandb-project arcops-cyber \
  --wandb-run-name teacher_v0_grpo
```

- `--use-sample-dataset` is available for dry runs.
- Add `--dry-run` to inspect the Hydra command without executing.

## 3. Teacher_v1 evaluation commands

```bash
# Runtime eval with memory on
python scripts/arcops_cyber/run_batch.py teacher --output paper_assets/arcops_cyber/results_teacher_v1.json

# Runtime eval with memory off (toggle via override)
python scripts/arcops_cyber/run_batch.py teacher \
  --output paper_assets/arcops_cyber/results_teacher_v1_memory_off.json \
  --scenarios scenario_1 scenario_2 scenario_3 scenario_4 scenario_5
```

Aggregated summaries are written alongside raw records; compare against the student baseline to compute the J-curve deltas.
