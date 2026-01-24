# ATLAS Quickstart Evaluation

Run a single question through the ATLAS loop to see how much a GPT-5 teacher and the Reward Interpretation System (RIM) improve your agent before you export traces and launch GRPO training.

## Prerequisites
- Python 3.11+
- `pip install atlas-core`
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

Use the optional flags to change models or token limits. When you’re ready to graduate from evaluation to training, export a batch of runtime traces with the atlas-sdk CLI and run:
```bash
atlas-core offline-pipeline --export-path traces/<your-export>.jsonl
```
This converts the export into a GRPO training job without any additional overrides.
