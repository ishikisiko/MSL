# Repository Guidelines

## Project Structure & Module Organization
- `MLS1/`: MNIST training, TFLite conversion, and microcontroller inference managed by the local `Makefile`.
- `MLS2/`: Multi-scale optimization scripts; checkpoints in `models/` and metrics in `results/`.
- `MLS3/`: Hardware-aware framework with `Makefile` automation, modules (`part1_baseline_model.py`, `part2_optimizations.py`, `performance_profiler.py`), and outputs in `optimized_models/`, `results/`, `logs/`.
- `Midterm/`: Static coursework site; root archives remain read-only references.

## Build, Test, and Development Commands
- Install deps with `python -m pip install -r MLS3/requirements.txt` or `python -m pip install -r MLS2/requirements.txt`.
- Drive MLS3 via `make -C MLS3 install|train|optimize|benchmark|analyze|clean`; append `PLATFORM=`/`SIMULATOR=` for deployments.
- Execute MLS2 scripts (`python MLS2/part2_cloud_optimization.py`, `python MLS2/inference_edge_device.py`, etc.) and capture outputs in `results/`.
- Build the quantized demo with `make -C MLS1` (debug/asan variants) and run `./mnist_inference`.

## Coding Style & Naming Conventions
- Python: follow PEP 8 with 4-space indents, docstrings, and type hints mirroring `MLS3/part1_baseline_model.py`; keep modules/functions `snake_case`, classes `PascalCase`.
- Run `make -C MLS3 lint` (flake8 + black --check) or `black MLS2/*.py` before committing; omit auto-generated notebook checkpoints.
- C++ in `MLS1/` uses 2-space indents, same-line braces, and `constexpr` globals as in `part3_micro_deployment.cc`.
- Logs stay concise; keep bilingual strings in MLS2 and add an English summary when extending them.

## Testing Guidelines
- Place pytest suites in `MLS3/tests/` (or module-level `tests/`) named `test_*.py`; run `make -C MLS3 test` or `python -m pytest MLS3/tests -v`.
- For MLS2, add smoke tests that stub hardware dependencies and write sample metrics to `results/`.
- Validate MLS1 changes with `./mnist_inference` on `mnist_5_samples.h` and note accuracy in the PR.
- Capture profiler output (`python MLS3/performance_profiler.py --validate`) whenever pipeline or deployment code changes.

## Commit & Pull Request Guidelines
- Write commits that are short, scope-prefixed, and action-oriented (e.g., `MLS3: refine quantization search`); reference issues when available.
- Group changes by assignment and avoid mixing code, assets, and docs in one commit.
- PRs should outline the change, list targets, note executed commands, and attach key logs or screenshots.
- Flag new binaries or configs, update `.gitignore` if required, and confirm reviewer approval before tracking additional models.

## Environment & Artifact Handling
- Keep secrets out of git; document required env vars in the relevant README.
- Regenerate models through the documented scripts and store them under `models/` or `optimized_models/` with semantic names.
- Rotate `logs/` into dated folders and avoid committing temporary sandbox transcripts or bulky intermediates.
