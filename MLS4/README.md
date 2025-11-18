# Model Compression and Optimization Pipeline

## Setup and Execution Instructions

### 1. Environment Setup
It is recommended to use a virtual environment to manage dependencies.

```bash
conda create -n tf220 python=3.10 -y
conda activate tf220

git clone https://github.com/ishikisiko/MSL.git
cd MSL/MLS4
# 3. 装 GPU 版 TF（会自动拉配套的 CUDA/cuDNN）
python -m pip install "tensorflow[and-cuda]"
# 安装 legacy 版 Keras（与 tf.keras 对应）
python -m pip install tf_keras
```

### 2. Install Dependencies
Install the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 3. Running the Pipeline
The main execution pipeline is orchestrated in `main.py`. This script performs:
1. Deterministic CIFAR-100 preparation (splits cached under `results/datasets/`).
2. Baseline EfficientNet-B0 training (30 epochs, AdamW/SGDW + weight decay + EMA).
3. Compression experiments (pruning, quantization, distillation).
4. Interaction analyses, evaluations, and visualization exports.

Key baseline controls are exposed via CLI flags:

| Flag | Description | Default |
| --- | --- | --- |
| `--train-baseline` | Forces baseline retraining prior to compression runs | _off_ |
| `--baseline-epochs` | Total training epochs (fixed to 30 per requirements) | `30` |
| `--baseline-batch-size` | Training batch size with fixed shapes/drop remainder | `256` |
| `--baseline-optimizer` | `adamw` or `sgdw` (both decoupled weight decay) | `adamw` |
| `--baseline-lr` | Base LR used for cosine-restart schedule | `6e-4` |
| `--baseline-weight-decay` | Weight decay for AdamW/SGDW | `1e-4` |
| `--ema-decay` / `--disable-ema` | EMA smoothing for evaluation weights | `0.999` |

Example end-to-end execution:

```bash
python main.py --train-baseline --baseline-optimizer adamw
```

All training histories are written to `results/baseline_training_history.json` and summarized in `reports/baseline_summary.json`, while evaluation benchmarks land in `results/pipeline_evaluations.json`.

The representative calibration set used for post-training quantization is persisted at `results/datasets/calibration_samples.npz`, ensuring Parts 2–4 consume identical inputs.

#### Quick Run on Google Colab
When prototyping in Colab, prefix shell commands with `!` (or `%cd` for directory changes). A minimal end-to-end session looks like:
```bash
!git clone --depth 1 https://github.com/ishikisiko/MSL temp_repo
!cp -r temp_repo/MLS4 . && rm -rf temp_repo
%cd MLS4
!pip install -r requirements.txt
!python main.py --train-baseline
```
Adjust the repo URL as needed and keep the `!` prefix to avoid Colab syntax errors.

### 4. Expected Output
- **Models**: The trained baseline model (`baseline_model.keras`) and all compressed model variants will be saved in the `compressed_models/` directory.
- **Checkpoints**: Training checkpoints will be saved in the `checkpoints/` directory.
- **Analysis Documents**: PDF reports, Markdown guidelines, and Excel benchmarks will be generated.
- **Visualizations**: PNG images of plots and heatmaps will be created.
