# Day 0 Execution Plan – ATLAS Core Technical Report

Owner: Jarrod Barnes  
Source of truth: `Foundational_Paper_Outline.md`  
Business context: `memo.md`

## Objectives
1. Materialise the ArcOps-Cyber dataset (ExCyTIn-derived) for runtime experiments and offline GRPO.
2. Configure atlas-sdk to run off-the-shelf GPT-5-mini (Student) / GPT-5 (Teacher) pairs via LiteLLM.
3. Stage automation and documentation so Day 1 can immediately capture traces and train `Teacher_v1`.

## Tasks

| Status | Category | Task | Notes |
| --- | --- | --- | --- |
| ☑ | Dataset | Review generated tasks for balance (questions per incident, context length) | Inspect `paper_assets/arcops_cyber/dataset_index.json` |
| ☑ | Dataset | Choose final evaluation subset (target 40–60 questions) and record IDs | Update `paper_assets/arcops_cyber/README.md` with selection |
| ☑ | Dataset | Define RIM scoring adapter for exact/fuzzy matching | Implemented in `scripts/arcops_cyber/scoring.py` |
| ☑ | Runtime | Point atlas run configs to `configs/examples/arcops_cyber_runtime.yaml` | Confirm environment variables via `.env` (Postgres re-enabled via `atlas init`) |
| ☑ | Runtime | Draft batch runner script for Student-only vs Student+Teacher seeds | Implemented `scripts/arcops_cyber/run_batch.py` |
| ☑ | Runtime | Smoke-test config on 2 sample questions (verify telemetry & exports) | Verified via `run_batch.py --limit 1`; Postgres container running for trace persistence. |
| ☑ | Offline | Prepare Hydra overrides for GRPO training on ArcOps-Cyber traces | Documented in `paper_assets/arcops_cyber/offline_commands.md` (uses `atlas train`). |
| ☑ | Offline | Document evaluation command set (memory on/off) | Added batch + memory-off commands to `paper_assets/arcops_cyber/offline_commands.md`. |

## Artifacts Created Today
- `scripts/arcops_cyber/prepare_arcops_cyber.py` – downloads and structures ArcOps-Cyber assets.
- `paper_assets/arcops_cyber/` – dataset snapshot with per-question JSON specs and provenance notes.
- `configs/examples/arcops_cyber_runtime.yaml` – LiteLLM-based GPT-5-mini (Student) / GPT-5 (Teacher) runtime harness.

## Next Checkpoint
Use this plan to drive the 48–72 hour execution window:
- Day 1: run dataset through atlas runtime, export traces, and schedule GRPO training.
- Day 2: generate figures/tables and draft sections 1–6.
- Day 3: finalize manuscript, appendices, and packaging.
