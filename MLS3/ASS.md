# Hardware-Aware Design Assignment

**Due:** Week 8

## Objective
Design and implement a hardware-aware machine learning application that considers energy, latency, and memory constraints across different deployment targets. Optimize a computer vision model for multiple hardware platforms, applying hardware-specific optimizations and analyzing performance trade-offs through either real hardware deployment or comprehensive simulation and modeling.

## Background
This assignment builds on the hardware acceleration principles discussed in weeks 6-8. You will experience how hardware characteristics influence ML system design, from algorithm selection to memory management and energy optimization. The work emphasizes the practical challenges of deploying AI at the edge while maintaining acceptable performance.

## Hardware Requirements Notice
This assignment can be completed entirely using simulation tools and performance modeling. Students with access to specialized hardware are encouraged to use it, but it is not required for full credit. You may choose between:

- **Track A:** Real Hardware Deployment — deploy on actual hardware platforms with real measurements.
- **Track B:** Simulation & Modeling — use simulation tools and theoretical analysis for comprehensive platform evaluation.

Both tracks can achieve full credit and meet all learning objectives.

---

## Part 1: Baseline Model Implementation (25 points)
Implement a MobileNetV2-based image classification model that will serve as the baseline for subsequent hardware-aware optimizations.

### Implementation Requirements
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import psutil
import os


def create_baseline_model(input_shape=(224, 224, 3), num_classes=10):
    """Create and compile a MobileNetV2-based classifier."""
    # TODO: Implement MobileNetV2 architecture
    # Base model: MobileNetV2 (pretrained on ImageNet, frozen initially)
    # Add custom classification head for your dataset
    # Include global average pooling and dropout for regularization
    pass


def load_and_preprocess_data():
    """Load CIFAR-10 and preprocess for MobileNetV2."""
    # TODO: Load CIFAR-10 dataset
    # Resize images to 224x224 for MobileNetV2
    # Normalize pixel values and apply data augmentation
    # Use tf.data.Dataset for efficient data loading
    pass


def benchmark_baseline_model(model, test_data, batch_size=32):
    """Benchmark the baseline model for latency, memory, and accuracy."""
    # TODO: Measure inference latency (single sample and batch)
    # TODO: Monitor memory usage during inference
    # TODO: Calculate FLOPs and model parameters
    # TODO: Measure energy consumption (if possible on your platform)

    metrics = {
        'single_inference_time': 0.0,
        'batch_inference_time': 0.0,
        'memory_usage_mb': 0.0,
        'model_size_mb': 0.0,
        'accuracy': 0.0,
        'flops': 0,
        'parameters': 0,
    }
    return metrics


if __name__ == "__main__":
    # Load data
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    # Create and train model
    model = create_baseline_model()

    # Train with fine-tuning strategy
    # 1. Train classification head only (base frozen)
    # 2. Unfreeze base model and fine-tune with lower learning rate

    # Save the trained model
    model.save('baseline_mobilenetv2.keras')

    # Benchmark performance
    metrics = benchmark_baseline_model(model, x_test)
    print("Baseline Model Performance:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
```

### Deliverables
- Complete baseline MobileNetV2 implementation.
- Training logs showing the transfer learning approach.
- Achieved test accuracy (target: >85% on CIFAR-10).
- Baseline performance metrics including latency, memory, and model size.

---

## Part 2: Hardware-Aware Optimizations (30 points)
Implement multiple optimization techniques targeting different hardware constraints.

### 2.1 Model Architecture Optimization
```python
def create_optimized_models():
    """Create hardware-optimized variants of the baseline model."""
    models = {}

    # TODO: Implement latency-optimized model
    # - Reduce MobileNetV2 depth multiplier
    # - Use depthwise separable convolutions
    # - Optimize for inference speed
    models['latency_optimized'] = create_latency_optimized_model()

    # TODO: Implement memory-optimized model
    # - Reduce model width (channels)
    # - Use model pruning techniques
    # - Optimize for memory footprint
    models['memory_optimized'] = create_memory_optimized_model()

    # TODO: Implement energy-optimized model
    # - Use quantization-aware training
    # - Reduce computation intensity
    # - Optimize for energy efficiency
    models['energy_optimized'] = create_energy_optimized_model()

    return models


def create_latency_optimized_model():
    """Create model optimized for inference latency."""
    # TODO: Use MobileNetV2 with alpha=0.5 (depth multiplier)
    # TODO: Reduce input resolution to 160x160 or 128x128
    # TODO: Use grouped convolutions where possible
    pass


def create_memory_optimized_model():
    """Create model optimized for memory usage."""
    # TODO: Implement model pruning (structured and unstructured)
    # TODO: Use knowledge distillation from baseline model
    # TODO: Reduce number of filters per layer systematically
    pass


def create_energy_optimized_model():
    """Create model optimized for energy consumption."""
    # TODO: Implement quantization-aware training
    # TODO: Use early exit mechanisms
    # TODO: Optimize activation functions for efficiency
    pass
```

### 2.2 Quantization Implementation
```python
def apply_quantization_optimizations(model, x_train_sample):
    """Apply different quantization techniques to the model."""
    quantized_models = {}

    # TODO: Post-training quantization (PTQ)
    quantized_models['ptq_int8'] = post_training_quantization(model, x_train_sample)

    # TODO: Quantization-aware training (QAT)
    quantized_models['qat_int8'] = quantization_aware_training(model, x_train_sample)

    # TODO: Mixed precision quantization
    quantized_models['mixed_precision'] = mixed_precision_quantization(model)

    # TODO: Dynamic range quantization
    quantized_models['dynamic_range'] = dynamic_range_quantization(model)

    return quantized_models


def representative_dataset_generator(x_train_sample):
    """Yield representative samples for quantization calibration."""
    # TODO: Provide 100-500 samples covering the distribution
    pass
```

### 2.3 Memory Management Optimization
```python
def implement_memory_optimizations(model):
    """Implement memory-aware optimizations."""
    optimizations = {}

    # TODO: Implement gradient checkpointing
    optimizations['gradient_checkpointing'] = implement_gradient_checkpointing(model)

    # TODO: Implement model sharding for large models
    optimizations['model_sharding'] = implement_model_sharding(model)

    # TODO: Implement activation compression
    optimizations['activation_compression'] = implement_activation_compression(model)

    # TODO: Optimize batch size for memory constraints
    optimizations['optimal_batch_size'] = find_optimal_batch_size(model)

    return optimizations
```

### Deliverables
- Implementation of all four optimization strategies.
- Quantized model variants with accuracy preservation analysis.
- Memory optimization techniques with measured improvements.
- Performance comparison across all optimized variants.

---

## Part 3: Multi-Platform Analysis (25 points)
Choose Track A **or** Track B based on your hardware access.

### Track A: Real Hardware Deployment
Deploy optimized models across different hardware platforms and measure real performance.

#### 3.1 Platform-Specific Optimization
```python
class HardwareOptimizer:
    """Hardware-aware model optimizer for different platforms."""

    def __init__(self, target_platform):
        self.platform = target_platform
        self.optimization_config = self._get_platform_config()

    def _get_platform_config(self):
        """Get optimization configuration for the target platform."""
        configs = {
            'cpu_x86': {
                'optimization_level': 'O3',
                'use_avx': True,
                'thread_count': 4,
                'memory_constraint_mb': 1024,
            },
            'arm_cortex_a': {
                'use_neon': True,
                'fp16_acceleration': True,
                'memory_constraint_mb': 512,
                'power_budget_mw': 2000,
            },
            'arm_cortex_m': {
                'quantization': 'int8',
                'memory_constraint_kb': 256,
                'power_budget_mw': 50,
                'use_cmsis_nn': True,
            },
            'gpu_mobile': {
                'use_gpu_delegate': True,
                'fp16_inference': True,
                'memory_constraint_mb': 2048,
                'thermal_throttling': True,
            },
        }
        return configs.get(self.platform, {})

    def optimize_for_platform(self, model):
        """Apply platform-specific optimization pipeline."""
        # TODO: CPU — Intel OpenVINO or ARM Compute Library optimizations
        # TODO: ARM Cortex-A — NEON SIMD instructions, FP16 execution
        # TODO: ARM Cortex-M — CMSIS-NN, aggressive quantization
        # TODO: Mobile GPU — GPU delegates, texture memory optimization
        pass


def deploy_to_tflite_micro(model, target_mcu='cortex_m4'):
    """Deploy model to TensorFlow Lite Micro for MCU targets."""
    pass


def deploy_to_mobile_gpu(model, target_gpu='adreno_640'):
    """Deploy model using mobile GPU acceleration."""
    pass


def deploy_to_edge_tpu(model):
    """Deploy model to Google Coral Edge TPU."""
    pass
```

#### Deliverables
- Working deployments on at least three different platform types.
- Platform-specific optimization implementations.
- Real performance measurements across all platforms.
- Stress testing results showing sustained performance.

### Track B: Simulation & Performance Modeling
Develop comprehensive performance models and validate them using simulation tools.

#### 3.1 Platform Performance Modeling
```python
class PlatformPerformanceModel:
    """Model performance characteristics across hardware platforms."""

    def __init__(self, platform_specs):
        self.platform_specs = platform_specs
        self.performance_models = self._build_performance_models()

    def _build_performance_models(self):
        """Build analytical performance models for each platform."""
        models = {}

        # TODO: Implement roofline models (memory bandwidth, compute throughput, cache hierarchy)
        models['roofline'] = self._build_roofline_model()

        # TODO: Implement energy models based on operation counts
        models['energy'] = self._build_energy_model()

        # TODO: Implement memory access models
        models['memory'] = self._build_memory_model()

        return models

    def estimate_performance(self, model_graph, platform_type):
        """Estimate performance metrics for a model on a platform."""
        # TODO: Analyze model graph for FLOPs and memory access patterns
        # TODO: Apply platform-specific performance models
        # TODO: Consider thermal throttling and power constraints
        pass


def validate_with_simulation(models, platform_configs):
    """Validate performance estimates using simulation tools."""
    validation_results = {}

    # TODO: Use QEMU for ARM simulation
    validation_results['qemu_arm'] = simulate_arm_performance(models)

    # TODO: Use Renode for Cortex-M simulation
    validation_results['renode_cm'] = simulate_cortex_m_performance(models)

    # TODO: Use WebGPU as mobile GPU proxy
    validation_results['webgpu_proxy'] = simulate_mobile_gpu_performance(models)

    # TODO: Cross-validate simulation results with analytical models
    validation_results['model_accuracy'] = cross_validate_models(validation_results)

    return validation_results


def simulate_arm_performance(models):
    """Simulate ARM Cortex-A performance using QEMU."""
    # TODO: Set up QEMU ARM environment
    # TODO: Deploy TFLite models and measure performance
    # TODO: Extract detailed performance metrics
    pass


def simulate_cortex_m_performance(models):
    """Simulate Cortex-M performance using Renode."""
    # TODO: Set up Renode simulation environment
    # TODO: Deploy TFLite Micro models
    # TODO: Measure cycle counts and memory usage
    pass


def simulate_mobile_gpu_performance(models):
    """Simulate mobile GPU performance using a WebGPU proxy."""
    # TODO: Convert models to WebGPU-compatible format
    # TODO: Measure performance in a browser environment
    # TODO: Scale results based on mobile GPU specifications
    pass
```

#### 3.2 Cross-Platform Analysis Framework
```python
class CrossPlatformAnalyzer:
    """Analyze optimization effectiveness across different platforms."""

    def __init__(self):
        self.platform_characteristics = self._load_platform_specs()

    def _load_platform_specs(self):
        """Load detailed specifications for target platforms."""
        return {
            'x86_desktop': {
                'core_count': 8,
                'frequency_ghz': 3.5,
                'simd_width': 256,  # AVX2
                'l1_cache_kb': 32,
                'l2_cache_kb': 256,
                'l3_cache_mb': 8,
                'memory_bandwidth_gbps': 50,
                'tdp_watts': 65,
            },
            'arm_cortex_a78': {
                'core_count': 4,
                'frequency_ghz': 2.4,
                'simd_width': 128,  # NEON
                'l1_cache_kb': 64,
                'l2_cache_kb': 512,
                'memory_bandwidth_gbps': 15,
                'tdp_watts': 5,
            },
            'arm_cortex_m7': {
                'frequency_mhz': 400,
                'sram_kb': 512,
                'flash_mb': 2,
                'power_budget_mw': 100,
                'has_fpu': True,
                'dsp_extensions': True,
            },
        }

    def analyze_optimization_effectiveness(self, baseline_model, optimized_models):
        """Analyze how different optimizations perform across platforms."""
        results = {}

        for platform in self.platform_characteristics:
            platform_results = {}

            # TODO: Model baseline performance on platform
            baseline_perf = self.model_performance(baseline_model, platform)

            # TODO: Model optimized variants performance
            for opt_name, opt_model in optimized_models.items():
                opt_perf = self.model_performance(opt_model, platform)
                platform_results[opt_name] = {
                    'speedup': baseline_perf['latency'] / opt_perf['latency'],
                    'memory_reduction': 1 - (opt_perf['memory'] / baseline_perf['memory']),
                    'energy_reduction': 1 - (opt_perf['energy'] / baseline_perf['energy']),
                    'accuracy_loss': baseline_perf['accuracy'] - opt_perf['accuracy'],
                }

            results[platform] = platform_results

        return results

    def model_performance(self, model, platform):
        """Model expected performance of a model on a platform."""
        # TODO: Implement detailed performance modeling
        # Consider instruction-level analysis, memory access patterns,
        # cache behavior, and thermal constraints
        pass
```

#### Deliverables
- Comprehensive performance models for three or more platform types.
- Simulation validation using multiple tools (QEMU, Renode, WebGPU).
- Cross-platform optimization effectiveness analysis.
- Detailed theoretical analysis with literature validation.

---

## Part 4: Comprehensive Analysis and Design Recommendations (20 points)

### 4.1 Performance Analysis
Create detailed performance comparisons and provide hardware-aware design recommendations.

```python
def generate_performance_report():
    """Generate comprehensive performance analysis report."""
    # TODO: Create performance comparison tables
    performance_data = {
        'models': ['baseline', 'latency_opt', 'memory_opt', 'energy_opt'],
        'platforms': ['cpu_x86', 'arm_cortex_a', 'arm_cortex_m', 'mobile_gpu'],
        'metrics': ['latency_ms', 'memory_mb', 'energy_mj', 'accuracy', 'throughput_fps'],
    }

    # TODO: Generate radar charts for multi-dimensional trade-offs
    # TODO: Create Pareto frontier analysis for accuracy vs efficiency
    # TODO: Analyze performance scaling with different batch sizes

    return performance_data


def analyze_hardware_utilization():
    """Analyze utilization of hardware features by each model."""
    utilization_analysis = {
        'cpu_simd_utilization': {},
        'memory_bandwidth_efficiency': {},
        'cache_efficiency': {},
        'thermal_characteristics': {},
        'power_efficiency': {},
    }

    return utilization_analysis
```

### 4.2 Design Methodology
```python
class HardwareAwareDesignMethodology:
    """Framework for hardware-aware ML system design."""

    def __init__(self):
        self.design_principles = [
            'co_design_hardware_software',
            'early_constraint_specification',
            'iterative_optimization',
            'multi_objective_optimization',
            'platform_specific_tuning',
        ]

    def constraint_specification_framework(self, application_requirements):
        """Specify hardware constraints early in the design."""
        # TODO: Define methodology for power, memory, latency, accuracy targets
        pass

    def optimization_priority_framework(self, constraints):
        """Determine optimization priorities based on constraints."""
        # TODO: Create decision tree for selecting optimization strategies
        pass

    def design_space_exploration(self, base_model, constraints):
        """Explore design alternatives systematically."""
        # TODO: Implement automated exploration of architectures, quantization levels, and techniques
        pass
```

### Required Analysis Report (3-4 pages)
Address the following topics:

1. **Hardware-Software Co-Design Analysis**
   - How did hardware constraints influence model architecture decisions?
   - What trade-offs emerged between accuracy and hardware efficiency?
   - How did different optimization techniques interact with hardware characteristics?
2. **Platform-Specific Optimization Insights**
   - Which optimizations were most effective for each platform type?
   - How did memory hierarchy differences impact optimization strategies?
   - What role did specialized hardware features (SIMD, tensor cores) play in performance?
3. **Energy-Latency-Accuracy Trade-off Analysis**
   - Create Pareto frontier plots showing trade-offs between metrics.
   - Analyze which applications benefit from each optimization approach.
   - Discuss implications for battery-powered edge devices.
4. **Scalability and Deployment Considerations**
   - How do optimizations scale across different hardware generations?
   - What challenges arise when deploying across heterogeneous hardware?
   - How would you handle model updates in resource-constrained environments?
5. **Design Methodology Recommendations**
   - Propose a systematic approach for hardware-aware ML system design.
   - Identify tools and frameworks that improve the design process.
   - Describe how to incorporate hardware constraints into the ML development lifecycle.
6. **Future Hardware Trends Impact**
   - How might emerging hardware trends affect design decisions?
   - What new optimization opportunities do you foresee?

---

## Submission Requirements

### Code Deliverables
- `part1_baseline_model.py` — baseline MobileNetV2 implementation.
- `part2_optimizations.py` — hardware-aware optimizations.
- `part3_deployment.py` (Track A) or `part3_modeling.py` (Track B) — platform analysis code.
- `performance_profiler.py` — performance measurement utilities.
- `optimization_framework.py` — hardware-aware design framework.
- `requirements.txt` — Python dependencies.
- `Makefile` or build scripts for deployments and simulations.

### Model Files
- `baseline_mobilenetv2.keras` — trained baseline model.
- `optimized_models/` — directory containing all optimized model variants.
- Platform-specific model files (`.tflite`, `.onnx`, etc.).

### Analysis Documents
- `hardware_aware_analysis.pdf` — comprehensive analysis report (3-4 pages).
- `performance_benchmarks.xlsx` — detailed performance measurements.
- `optimization_guidelines.md` — design methodology documentation.

### Demo Materials
- `README.md` — setup and execution instructions.
- `demo_notebook.ipynb` — Jupyter notebook demonstrating key results.
- Platform-specific demo scripts.
- Optional: video demonstration (3-5 minutes).

---

## Hardware Access and Simulation Resources

### Track A: Recommended Hardware Platforms
- CPU: Intel/AMD x86, ARM Cortex-A series.
- Mobile GPU: Qualcomm Adreno, ARM Mali, Apple GPU.
- Edge AI: Google Coral Dev Board, Intel Neural Compute Stick.
- MCU: Arduino Nano 33 BLE, STM32 Discovery boards.

### Track B: Simulation Alternatives
- ARM Cortex-A (Linux-class): QEMU (system or user mode) running ARM binaries (e.g., TFLite/ONNX Runtime). Label results as `[QEMU-ARM]`.
- ARM Cortex-M (MCU-class): Renode (preferred) or QEMU Cortex-M running TensorFlow Lite Micro firmware with INT8 models. Label results as `[Renode-CM]` or `[QEMU-CM]`.
- Mobile GPU proxy: ONNX Runtime Web with WebGPU in a Chromium-based browser, or ncnn (Vulkan) on desktop as a proxy for mobile GPUs. Label results as `[WebGPU-proxy]` or `[Vulkan-proxy]`.
- x86 CPU: Native desktop/laptop execution using TFLite or ONNX Runtime for baseline comparisons. Label results as `[native-x86]`.

#### Notes for Track B
- Clearly tag all reported metrics with the execution context.
- Document versions, commands, and flags so results are reproducible.
- Treat simulated/proxy results as relative comparisons; discuss limitations.
- Label estimated numbers explicitly and describe estimation methods.
- Prefer empirical measurements (even in simulators) over purely theoretical calculations.

### Performance Estimation Tools (Both Tracks)
- TensorFlow Model Optimization Toolkit (TF-MOT) for pruning/quantization analysis.
- `tflite_benchmark_model` and `onnxruntime_perf_test` for repeatable latency tests.
- FLOPs/parameter estimators: `tensorflow.python.profiler`, `keras-flops`, or Netron.
- Power/energy tools: `powertop`, `turbostat` (Linux) for package power estimation.
- Published benchmarks for comparative context (cite sources).

---

## Rubric

### Baseline Implementation (25 pts)
| Rating | Description |
| --- | --- |
| 25 to >23 pts (Excellent) | Robust MobileNetV2 implementation, >90% accuracy, comprehensive benchmarking with detailed metrics. |
| 23 to >20 pts (Good) | Working implementation, >85% accuracy, good benchmarking with most key metrics. |
| 20 to >15 pts (Satisfactory) | Basic implementation, >80% accuracy, limited benchmarking with basic metrics. |
| 15 to >0 pts (Needs Improvement) | Implementation issues, poor accuracy (<80%), or missing benchmarking. |
| 0 pts (No Marks) | Empty submission. |

### Hardware Optimizations (30 pts)
| Rating | Description |
| --- | --- |
| 30 to >27 pts (Excellent) | All four optimization types implemented with significant improvements and thorough analysis. |
| 27 to >24 pts (Good) | Three to four optimizations working well with solid improvements and adequate analysis. |
| 24 to >19 pts (Satisfactory) | Two to three optimizations working with moderate improvements and basic analysis. |
| 19 to >0 pts (Needs Improvement) | Limited optimizations, poor results, or minimal analysis. |
| 0 pts (No Marks) | Empty submission. |

### Multi-Platform Analysis (25 pts)
| Rating | Description |
| --- | --- |
| 25 to >23 pts (Excellent) | Track A: deployment on 3+ platforms with comprehensive measurements. Track B: modeling across 3+ platforms with validated estimates and cross-validation. |
| 23 to >20 pts (Good) | Track A: 2-3 platforms with good measurements. Track B: good modeling of 2-3 platforms with validation. |
| 20 to >16 pts (Satisfactory) | Track A: two platforms with basic measurements. Track B: basic modeling of two platforms with limited validation. |
| 16 to >0 pts (Needs Improvement) | Track A: single platform or poor measurements. Track B: inadequate modeling or no validation. |
| 0 pts (No Marks) | Empty submission. |

### Performance Analysis & Design Methodology (20 pts)
| Rating | Description |
| --- | --- |
| 20 to >18 pts (Excellent) | Comprehensive analysis with actionable insights, excellent trade-off understanding, and practical design framework. |
| 18 to >16 pts (Good) | Good analysis with useful insights, solid understanding, and well-documented methodology. |
| 16 to >13 pts (Satisfactory) | Basic analysis with some insights and adequate documentation. |
| 13 to >0 pts (Needs Improvement) | Limited or superficial analysis with weak methodology. |
| 0 pts (No Marks) | Empty submission. |

**Total Points:** 100
