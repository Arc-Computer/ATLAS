# ArcOps-Cyber Dataset (Day 0 Snapshot)

ArcOps-Cyber is the security-operations slice that underpins the ATLAS Core technical report. It is derived from the public ExCyTIn-Bench dataset released by Microsoft and provides a reproducible set of incident-response questions for evaluating the atlas-sdk runtime and offline GRPO loop.

## Provenance

- **Source**: https://huggingface.co/datasets/anandmudgerikar/excytin-bench  
- **Incidents**: `incident_5`, `incident_34`, `incident_38`, `incident_39`, `incident_55`, `incident_134`, `incident_166`, `incident_322`  
- **Splits**: both `train/` and `test/` question sets are mirrored locally.  
- **Licence**: CDLA-Permissive 2.0 (matches upstream release).

The helper script `scripts/arcops_cyber/prepare_arcops_cyber.py` downloads the upstream JSON files, stores them under `raw/`, and emits per-question task specs under `tasks/`. Each task spec captures:

```json
{
  "task_id": "incident_5_test_q001",
  "incident": "incident_5",
  "context": "...",
  "question": "...",
  "gold_answer": "...",
  "solution_steps": ["..."],
  "metadata": {
    "start_alert": 14,
    "end_alert": 11
  }
}
```

## Day 0 Deliverables

1. **Dataset assets** – `raw/`, `tasks/`, and `dataset_index.json` for reproducibility.  
2. **Runtime integration** – task specs will drive atlas-sdk runs using off-the-shelf GPT‑5-mini (Student) and GPT‑5 (Teacher) via LiteLLM.  
3. **Offline loop inputs** – exported runtime traces will later feed GRPO to produce `Teacher_v1`.

Re-run the preparation script whenever the upstream dataset updates:

```bash
python scripts/arcops_cyber/prepare_arcops_cyber.py --force
```

## Scenario Splits (Day 0)

Scenario selection focuses on five incident progressions for runtime efficiency analysis:

- **Scenario 1** – Incident 38 (11 questions, fileless attack warm-up)
- **Scenario 2** – Incident 134 (12 questions, business email compromise)
- **Scenario 3** – Incident 5 (12 questions, Lockbit ransomware campaign)
- **Scenario 4** – Incident 166 (12 questions, SAP manipulation)
- **Scenario 5** – Incident 55 (12 questions, ADFS key exfiltration)

The file `scenario_splits.json` enumerates the exact task IDs for reproducibility.

## Runtime Notes

- Start Postgres once via `python -m atlas.cli.main init --compose-file paper_assets/atlas-postgres.yaml` (writes the compose file and boots the container on port 5433).
- Both `configs/examples/arcops_cyber_student.yaml` and `configs/examples/arcops_cyber_runtime.yaml` point at this database so Student-only and Teacher runs persist traces for GRPO.
- Batch runs: `python scripts/arcops_cyber/run_batch.py student --scenarios scenario_1 --limit 10 --output paper_assets/arcops_cyber/student_run.json` (switch `student` to `teacher` for guided runs). The output now includes per-scenario averages (success rate, latency, tokens).
