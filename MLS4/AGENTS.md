# Repository Guidelines

## Project Structure & Module Organization
The root contains targeted modules for each compression stage (`baseline_model.py`, `part1_pruning.py`, `part2_quantization.py`, `part3_distillation.py`, `part4_interaction_analysis.py`) plus shared utilities such as `compression_evaluator.py` and `visualization_tools.py`. Keep orchestration logic in `main.py` (create or extend as needed) so individual scripts remain single-purpose. Persist generated artifacts under `compressed_models/`, intermediate checkpoints under `checkpoints/`, and plots or tables in `visualizations/` or `reports/`.

## Build, Test, and Development Commands
- `python -m venv venv && source venv/bin/activate`: isolate dependencies before editing.
- `pip install -r requirements.txt`: install the exact versions expected by the grading harness.
- `python main.py`: run the full CIFAR-100 pipeline (data prep → training → pruning/quantization/distillation → evaluation).
- `python part1_pruning.py` / `python part2_quantization.py` / `python part3_distillation.py`: execute a single compression stage when iterating quickly.
- `python visualization_tools.py --input results/metrics.csv`: regenerate plots after updating evaluations.
- `pytest tests -v`: run unit and smoke tests; add focused suites per module under `tests/`.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, descriptive docstrings, and explicit type hints mirroring the existing modules. Keep functions and modules in `snake_case`, classes in `PascalCase`, and constants in `UPPER_SNAKE_CASE`. Before pushing, run `python -m black . --check` and `python -m flake8` to enforce formatting and linting.

## Testing Guidelines
New logic should arrive with `test_*.py` companions inside `tests/`, organized to mirror the module tree (for example, `tests/test_part2_quantization.py`). Use `pytest` fixtures to stub heavy training loops and to write sample metrics into `results/` so downstream scripts can be validated without full retraining. Snapshot key evaluation numbers (top-1 accuracy, compression ratio) and assert they stay within agreed tolerances.

## Commit & Pull Request Guidelines
Commits must be scoped and prefixed (e.g., `MLS4: tighten pruning schedule`, `Docs: clarify evaluator inputs`). In pull requests, summarize the change, link any assignment issue, list commands/tests executed, and attach representative logs or plots from `visualizations/`. Call out new artifacts (models, CSV metrics) and update `.gitignore` if temporary files are introduced. Seek review before tracking large binaries or checkpoints.

## Artifacts & Configuration
Do not commit secrets or raw datasets; document required environment variables in `README.md` when a path is needed. Regenerate models through the published scripts rather than hand-editing assets, and give outputs semantic filenames such as `compressed_models/resnet50_pruned_topk75.keras`. Archive dated logs under `logs/2024-*/` to keep the workspace tidy and prune obsolete artifacts once they are documented.
