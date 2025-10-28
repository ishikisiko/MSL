
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Hardware-Aware Design Assignment for MLS3 course that demonstrates optimization of computer vision models (MobileNetV2) across multiple hardware platforms with considerations for energy, latency, and memory constraints.

## Development Environment Setup

### Conda Environment for Modal Code
When running Modal code, always ensure you're in the "env2" conda environment:
```bash
eval "$(conda shell.bash hook)" && conda activate env2
```

### Modal Execution Commands
To run Modal files:
```bash
eval "$(conda shell.bash hook)" && conda activate env2 && python -m modal run XXX.py
```

To check Modal app logs after execution:
```bash
python -m modal app logs [APP_ID]
```

Note: On Windows, always use `python -m modal` instead of `modal` directly.

## Common Development Commands

### Training and Optimization
```bash
# Train baseline model
make train
python part1_baseline_model.py

# Run hardware-aware optimizations (includes fine-tuning)
make optimize
python part2_optimizations.py

# Run complete optimization pipeline
python run_optimizations.py
```

### Testing and Benchmarking
```bash
# Benchmark all models
make benchmark
python performance_profiler.py

# Run tests
make test
python -m pytest tests/ -v
```

### Deployment and Analysis
```bash
# Track A: Real hardware deployment
make deploy PLATFORM=cpu_x86    # Intel/AMD desktop
make deploy PLATFORM=arm_cortex_a  # Mobile/embedded processors
make deploy PLATFORM=arm_cortex_m  # Microcontrollers
make deploy PLATFORM=mobile_gpu   # Qualcomm Adreno, ARM Mali

# Track B: Simulation
make simulate SIMULATOR=qemu     # ARM Cortex-A simulation
make simulate SIMULATOR=renode   # ARM Cortex-M simulation
make simulate SIMULATOR=webgpu   # Mobile GPU proxy simulation

# Generate analysis reports
make analyze
python optimization_framework.py --generate-report
```

### Code Quality
```bash
# Format code
make format
black *.py

# Check code quality
make lint
flake8 *.py
black --check *.py
```

## Architecture Overview

### Core Components

1. **Baseline Model (`part1_baseline_model.py`)**: MobileNetV2-based CIFAR-10 classifier with two-phase training
   - Phase 1: Train classification head with frozen backbone (50 epochs)
   - Phase 2: Fine-tune last 50 layers (30 epochs)
   - Automatic GPU/CPU detection and multi-GPU support

2. **Hardware-Aware Optimizations (`part2_optimizations.py`)**: Creates three optimized variants
   - Latency-optimized: Reduced depth, smaller input
   - Memory-optimized: Reduced channel width, supports pruning
   - Energy-optimized: Balanced design for power efficiency

3. **Performance Profiler (`performance_profiler.py`)**: Comprehensive measurement tools
   - Latency metrics (mean, median, P95, P99)
   - Memory usage tracking
   - Energy estimation
   - Throughput analysis

4. **Optimization Framework (`optimization_framework.py`)**: Structured methodology
   - HardwareAwareDesignMethodology for constraint specification
   - PerformancePredictor for cross-platform modeling
   - Validation system with regression detection

5. **Dual Deployment System**:
   - Track A (`part3_deployment.py`): Real hardware deployment with TFLite conversion
   - Track B (`part3_modeling.py`): Simulation using QEMU, Renode, WebGPU

### Key Design Patterns

- **Automatic Environment Adaptation**: Code detects GPU availability and adjusts strategy
- **Multi-Objective Optimization**: Balances latency, memory, energy, accuracy
- **Two-Phase Training**: Frozen backbone → fine-tuning approach
- **Quantization Support**: PTQ, QAT, dynamic range, mixed precision

## Dependencies

Core requirements from `requirements.txt`:
- numpy==1.26.4
- tensorflow==2.15.1
- tensorflow-model-optimization==0.8.0
- psutil>=5.9.0
- matplotlib>=3.7.0

Developer tools:
- black==24.4.2
- flake8==7.1.1

## Important Notes

- TensorFlow logging is reduced to minimize noise (TF_CPP_MIN_LOG_LEVEL=2)
- GPU memory growth is enabled to prevent OOM errors
- Batch size is set to 256 for efficient GPU utilization
- Models are saved as .keras files in the root directory
- The project supports both local execution and Google Colab (see COLAB.md)

## Testing Strategy

- Use performance_profiler.py for comprehensive benchmarking
- Run `make validate` to check performance regression
- Each optimization variant should be validated against baseline
- Quantization models should be tested for accuracy degradation

## File Structure Notes

- Main models are saved in project root (e.g., `baseline_mobilenetv2.keras`)
- Optimized models go in `optimized_models/` directory
- Performance results stored in `results/` directory
- Logs saved in `logs/` directory
- Platform-specific configs in `platform_configs/`