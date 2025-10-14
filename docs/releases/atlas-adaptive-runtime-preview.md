# Atlas SDK Release Notes — v0.1.3
_Released: October 14, 2025_

## Overview

This release introduces Atlas as a continual-learning harness that turns every task into a structured learning episode. The SDK now routes requests through an adaptive runtime, ensuring your agents stay fast on familiar work while escalating to supervised execution for complex or unfamiliar tasks.

**Key Highlights:**
- Continual learning episodes with adaptive routing (auto → paired → coach → escalate)
- Persistent memory system that tags and reuses guidance from production
- Production-ready telemetry and structured data export for downstream training

<github-card url="https://github.com/Arc-Computer/atlas-sdk" />

---

## What's New in v0.1.3

**Adaptive Runtime**

The core runtime now includes four execution modes chosen per request by a capability probe. Lane enforcement ensures the teacher only intervenes where necessary, keeping routing latency minimal while biasing toward autonomous execution when history is strong.

- Four execution modes: `auto`, `paired`, `coach`, `escalate`
- Capability probe with lean payload (`{mode, confidence}`)
- Lane-specific orchestration for graduated supervision

**Learning & Memory**

Atlas now saves agent instructions and guidance from each run, tagged by reward signals (helpful vs. harmful). This persistent memory can be automatically reused in future tasks, creating a compounding knowledge base.

- Persistent memory tagged by reward
- Session-level evaluator that records final rewards and learning summaries
- Optional Postgres persistence for production scale (runs without database for quick trials)

**Developer Experience**

Bring your own agent configuration with support for Python, HTTP, and OpenAI-compatible endpoints. The `atlas` CLI provides scaffolding for triage adapters and local persistence, with lightweight defaults that keep your first run fast.

- Bring-your-own-agent configs (Python, HTTP, OpenAI-compatible)
- Structured YAML prompts for student and teacher personas
- `atlas` CLI scaffolds (`atlas triage init`, storage helpers)
- Opt-in advanced features (storage, exporters)

**Telemetry & Export**

Console telemetry streams adaptive mode, probe confidence, certification flags, and session summaries in real-time. JSONL export captures the full learning trajectory—plans, step attempts, guidance notes, adaptive history, and reward payloads—ready for downstream analytics or training pipelines.

- Real-time console telemetry for adaptive signals
- Structured JSONL export for training pipelines
- Validation results and certification tracking

**Documentation**

Updated quickstart guides explain how to register an agent, wire in triage adapters, and interpret the adaptive telemetry. New diagrams describe the triage → probe → lane → reward loop so you can align the architecture with your production workflows.

- Quickstart guides for agent registration and triage adapters
- Architecture diagrams for the triage → probe → lane → reward loop
- Telemetry interpretation and debugging guides

---

## Getting started

1. **Install**

   ```bash
   pip install arc-atlas
   ```

2. **Generate a triage adapter (optional but recommended)**

   ```bash
   atlas triage init --domain code --output triage_adapter.py
   ```

3. **Configure your agent**

   ```yaml
   agent:
     type: openai           # or "python", "http_api"
     name: demo-agent
     system_prompt: |
       You are the Atlas Student. Follow instructions carefully.
     tools: []
     llm:
       provider: openai
       model: gpt-5
       api_key_env: OPENAI_API_KEY

   adaptive_teaching:
     enabled: true
     certify_first_run: true
     triage_adapter: triage_adapter.build_dossier  # optional custom adapter
   ```

4. **Run an adaptive episode**

   ```python
   from atlas import core

   result = core.run(
       task="Investigate intermittent 500 errors during checkout",
       config_path="atlas_quickstart.yaml",
       stream_progress=True,
   )

   print(result.final_answer)
   ```

5. **Optional extras**
   - Enable storage (`atlas storage up`) when you need persona promotion history.
   - Export structured traces:

     ```bash
     python -m atlas.cli.export \
       --database-url postgresql://user:password@localhost:5432/atlas \
       --output session_traces.jsonl
     ```

---

## Looking ahead

- **Smarter routing**: richer probe inputs so we can explain *why* a lane was chosen and suggest relevant personas.
- **Granular retry controls** so supervised lanes can adopt lane-specific budgets over time, exposed via configuration.
- **Packaging improvements** (extras for storage/telemetry) to keep the base install lean.
- **Unified telemetry integrations** (OTEL publishers, default dashboards) so adaptive signals land in standard observability stacks.

---

## Feedback & contributions

Questions, bug reports, or ideas? Open an issue, send a PR, or reach us at [agent@arc.computer](mailto:agent@arc.computer). We'd love to hear how you're using the Atlas learning harness in production.
