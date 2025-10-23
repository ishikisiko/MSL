# Hardware-Aware Design Assignment

This repository contains the implementation for the Hardware-Aware Design Assignment, focusing on optimizing computer vision models across multiple hardware platforms with considerations for energy, latency, and memory constraints.

## Assignment Overview

The assignment consists of four main parts:

1. **Baseline Model Implementation** - MobileNetV2-based image classification
2. **Hardware-Aware Optimizations** - Multiple optimization techniques for different constraints
3. **Multi-Platform Analysis** - Either real hardware deployment (Track A) or simulation (Track B)
4. **Comprehensive Analysis** - Performance analysis and design recommendations

## Project Structure

```
├── part1_baseline_model.py      # Baseline MobileNetV2 implementation
├── part2_optimizations.py       # Hardware-aware optimizations
├── part3_deployment.py          # Track A: Real hardware deployment
├── part3_modeling.py            # Track B: Simulation & performance modeling
├── performance_profiler.py      # Performance measurement utilities
├── optimization_framework.py    # Hardware-aware design framework
├── requirements.txt             # Python dependencies
├── Makefile                     # Build automation
├── optimized_models/            # Directory for optimized model variants
├── logs/                        # Training and execution logs
├── results/                     # Performance results and benchmarks
└── platform_configs/            # Platform-specific configurations
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Git
- (Track A) Access to target hardware platforms (optional)
- (Track B) Simulation tools: QEMU, Renode (see requirements.txt)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MLS3
```

2. Install dependencies:
```bash
make install
# or manually:
pip install -r requirements.txt
```

3. Set up directory structure:
```bash
make setup
```

## Usage

### Track A: Real Hardware Deployment

For deployment on actual hardware platforms:

```bash
# Train baseline model
make train

# Apply optimizations
make optimize

# Deploy to specific platform
make deploy PLATFORM=cpu_x86
make deploy PLATFORM=arm_cortex_a
make deploy PLATFORM=arm_cortex_m
make deploy PLATFORM=mobile_gpu

# Run complete pipeline
make pipeline-a
```

### Track B: Simulation & Modeling

For simulation-based analysis:

```bash
# Train baseline model
make train

# Apply optimizations
make optimize

# Run simulations
make simulate SIMULATOR=qemu
make simulate SIMULATOR=renode
make simulate SIMULATOR=webgpu

# Run complete pipeline
make pipeline-b
```

### General Commands

```bash
# Benchmark all models
make benchmark

# Generate analysis reports
make analyze

# Run tests
make test

# Check code quality
make lint

# Format code
make format

# Clean generated files
make clean
```

## Platform Support

### Track A: Supported Hardware Platforms

- **CPU x86**: Intel/AMD desktop and server processors
- **ARM Cortex-A**: Mobile and embedded processors (Linux-class)
- **ARM Cortex-M**: Microcontrollers (MCU-class)
- **Mobile GPU**: Qualcomm Adreno, ARM Mali, Apple GPU

### Track B: Simulation Targets

- **QEMU**: ARM Cortex-A simulation (Linux-class)
- **Renode**: ARM Cortex-M simulation (MCU-class)
- **WebGPU**: Mobile GPU proxy simulation

## Model Variants

The project generates multiple optimized model variants:

- `baseline`: Original MobileNetV2 model
- `latency_optimized`: Optimized for inference speed
- `memory_optimized`: Optimized for memory footprint
- `energy_optimized`: Optimized for energy efficiency

### Quantization Variants

- PTQ (Post-Training Quantization)
- QAT (Quantization-Aware Training)
- Mixed Precision
- Dynamic Range

## Performance Metrics

Each model is evaluated across multiple dimensions:

- **Latency**: Single and batch inference time
- **Memory**: Model size and runtime memory usage
- **Energy**: Power consumption estimation
- **Accuracy**: Classification performance
- **Throughput**: Frames per second

## Analysis Reports

The framework generates comprehensive analysis including:

- Performance comparison tables
- Pareto frontier analysis
- Hardware utilization metrics
- Cross-platform optimization effectiveness
- Design methodology recommendations

## Dependencies

Key libraries and tools:

- TensorFlow/Keras for model development
- TensorFlow Lite for deployment
- ONNX for cross-platform compatibility
- QEMU/Renode for simulation
- Various profiling tools for performance measurement

## Contributing

This project follows standard Python development practices:

- Use `black` for code formatting
- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Add unit tests for new functionality

## License

This project is part of academic coursework and follows educational use guidelines.

## Support

For issues related to:

- **Setup and installation**: Check the Makefile targets and requirements.txt
- **Platform-specific deployment**: Refer to platform configuration files
- **Simulation setup**: Consult the simulation tool documentation
- **Performance analysis**: Review the profiler module documentation

## Deliverables

Upon completion, the project generates:

1. **Code Files**: All Python implementations
2. **Model Files**: Trained and optimized models
3. **Analysis Documents**: Performance reports and methodology
4. **Demo Materials**: Notebooks and demonstration scripts

## Timeline

- **Week 1-2**: Baseline implementation and training
- **Week 3-4**: Hardware-aware optimizations
- **Week 5-6**: Multi-platform analysis
- **Week 7-8**: Comprehensive analysis and reporting