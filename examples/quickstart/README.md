# ATLAS Quickstart Evaluation

Run a single question through the ATLAS loop to see how much a GPT-5 teacher and the RIM reward system improve your agent before committing to a full GEPA optimization run.

## Prerequisites
- Python 3.11+
- `pip install -r requirements-py312.txt`
- Environment variables:
  - `OPENAI_API_KEY`
  - `GEMINI_API_KEY`

## Usage
```bash
python examples/quickstart/evaluate.py \
  --question "Masha braided her dolls' hair..." \
  --teacher-model gpt-5 \
  --student-model gpt-4o-mini
```
The script prints baseline and teacher-guided responses plus their RIM scores.

Use the optional flags to change models or token limits. When ready for full prompt evolution, run:
```bash
./scripts/openai_agent_atlas.sh configs/wrappers/openai_existing_agent.yaml
```
