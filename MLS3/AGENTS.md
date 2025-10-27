# Repository Guidelines

## Project Structure & Module Organization
The repository spans multiple labs. `MLS1/` handles MNIST training, TFLite conversion, and microcontroller inference via its local `Makefile`. `MLS2/` contains multi-scale optimization scripts, storing checkpoints under `models/` and metrics in `results/`. `MLS3/` provides the hardware-aware pipeline with key modules (`part1_baseline_model.py`, `part2_optimizations.py`, `performance_profiler.py`) plus automation through `make`. The `Midterm/` directory is a static coursework reference and should remain untouched.

## Build, Test, and Development Commands
Install dependencies with `python -m pip install -r MLS3/requirements.txt` (or `MLS2/requirements.txt` for the optimization suite). Drive MLS3 workflows using `make -C MLS3 install`, `make -C MLS3 train`, `make -C MLS3 optimize`, `make -C MLS3 benchmark`, and `make -C MLS3 analyze`; append `PLATFORM=` or `SIMULATOR=` for target-specific runs. Build the quantized MNIST demo via `make -C MLS1` (use `debug` or `asan` targets as needed) and execute `./mnist_inference`. Run MLS2 scripts directly, e.g., `python MLS2/part2_cloud_optimization.py`, capturing outputs under `results/`.

## Coding Style & Naming Conventions
Follow PEP 8 in Python with 4-space indents, descriptive docstrings, and typing consistent with `MLS3/part1_baseline_model.py`. Keep functions and modules in `snake_case`, classes in `PascalCase`. Use `make -C MLS3 lint` (flake8 + `black --check`) or `black MLS2/*.py` before submitting changes. C++ code in `MLS1/` uses 2-space indents, same-line braces, and `constexpr` globals, mirroring `part3_micro_deployment.cc`.

## Testing Guidelines
Place pytest suites in `MLS3/tests/` (named `test_*.py`) and run `make -C MLS3 test` or `python -m pytest MLS3/tests -v`. For MLS2, author smoke tests that stub hardware interfaces and emit sample metrics to `results/`. Validate MLS1 adjustments with `./mnist_inference` on `mnist_5_samples.h` and record accuracy in review notes. Trigger `python MLS3/performance_profiler.py --validate` whenever deployment or pipeline code changes.

## Commit & Pull Request Guidelines
Craft commits that are short, scope-prefixed, and action-oriented (e.g., `MLS3: refine quantization search`). Group related changes and avoid mixing docs, assets, and code in a single commit. PRs should summarize the change, list affected targets, note executed validation commands, and attach key logs or screenshots. Flag new binaries or configs, update `.gitignore` when introducing generated artifacts, and secure reviewer approval before tracking additional models.

## Environment & Artifact Handling
Never commit secrets; document required environment variables in the relevant README. Regenerate models using the provided scripts and store them in `models/` or `optimized_models/` with semantic versioning. Rotate `logs/` into dated folders and exclude transient notebooks, sandbox transcripts, and bulky intermediates from version control.
