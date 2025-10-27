# Dependency Guide

This project now relies on a slim, TensorFlow-centric dependency stack. The
`requirements.txt` file lists the exact packages to install; the table below
summarises the scope and rationale for each entry.

| Package | Version | Purpose |
| --- | --- | --- |
| `numpy` | 1.25.2 | Matches TensorFlow 2.15 wheels and keeps TFLite converters stable. |
| `tensorflow` | 2.15.1 | Provides `tf.keras` APIs, TFLite converter, and profiling utilities used throughout `part1`–`part3`. |
| `tensorflow-model-optimization` | 0.8.0 | Supplies post-training and quantization-aware tooling leveraged in `part2_optimizations.py`. |
| `psutil` | ≥5.9 | Required by `performance_profiler.py` for memory telemetry. |
| `matplotlib` | ≥3.7 | Enables optional diagram generation in `generate_colab_diagrams.py`. |
| `black` | 24.4.2 | Used by `make lint`/`make format` for code formatting. |
| `flake8` | 7.1.1 | Complements `black` in `make lint` for static analysis. |

## Installation Tips

### Windows / Local

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

TensorFlow 2.15 automatically installs the correct Windows wheel (via
`tensorflow-intel`). If a GPU-enabled build is required, follow the official
CUDA installation instructions before installing TensorFlow.

### Google Colab

Colab already ships with CUDA-enabled TensorFlow. Running the same
`pip install -r requirements.txt` command upgrades NumPy to the pinned version
and installs the optimisation tooling. The install step takes ~2 minutes on a
fresh runtime.

### Verifying the Environment

After installation you can confirm the environment with:

```bash
python -c "import tensorflow as tf, numpy as np, psutil; print(tf.__version__, np.__version__)"
```

Optionally run `python quick_test.py` to ensure every project module imports
cleanly.
