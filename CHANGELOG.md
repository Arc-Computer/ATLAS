# Changelog

## 2025-10-11

### Removed
- Online optimization entrypoints (`optimize_teaching.py`, runtime adapters, optimize shell scripts, and configs/optimize assets).
- Online optimization notebooks and Mintlify pages that referenced the deprecated workflow.

### Added
- `scripts/run_offline_pipeline.py` helper for one-touch export → GRPO training workflows.
- GRPO-focused example configs (`configs/examples/quickstart.yaml`, `configs/demo/runtime_grpo.yaml`) aligned with runtime trace exports.
- Conditional dependency handling for `bitsandbytes` (Linux/CUDA only) and explicit `litellm` requirement so the reward tests install cleanly on CPU dev machines.

### Changed
- Repositioned Atlas Core documentation and README to describe the repo as offline GRPO + reward tooling, directing online continual learning to the atlas-sdk runtime.
- Updated Hydra defaults and configs to use runtime trace datasets and GRPO training out of the box.
- Refreshed docs navigation and quickstarts to highlight the export → train → deploy pipeline.
