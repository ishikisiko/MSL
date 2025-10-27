"""Performance profiling utilities for hardware-aware model optimization.

This module provides comprehensive performance measurement tools for analyzing
ML models across different hardware platforms, including latency, memory usage,
energy consumption, and computational complexity metrics.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import psutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2_as_graph,
)

# Suppress TensorFlow logging to avoid TensorSpec spam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def calculate_model_metrics(
    model: keras.Model, batch_size: int = 1
) -> Dict[str, float]:
    """Calculate comprehensive model metrics including FLOPs, parameters, and size.
    
    Args:
        model: Keras model to analyze
        batch_size: Batch size for FLOPs calculation
        
    Returns:
        Dictionary containing model metrics
    """
    metrics = {}
    
    # Model parameters
    metrics["parameters"] = float(model.count_params())
    
    # Model size in MB
    size_bytes = 0
    for weight in model.weights:
        dtype = np.dtype(str(weight.dtype))
        size_bytes += np.prod(weight.shape) * dtype.itemsize
    metrics["model_size_mb"] = size_bytes / (1024 ** 2)
    
    # FLOPs calculation - suppress TensorFlow logging to avoid TensorSpec spam
    try:
        # Temporarily suppress TensorFlow logging
        old_verbosity = tf.compat.v1.logging.get_verbosity()
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        
        inputs = tf.TensorSpec([batch_size, *model.input_shape[1:]], tf.float32)
        concrete_fn = tf.function(model).get_concrete_function(inputs)
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_fn)
        
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name="")
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(
                graph=graph, run_meta=run_meta, cmd="op", options=opts
            )
            metrics["flops"] = float(flops.total_float_ops if flops is not None else 0)
        
        # Restore original logging verbosity
        tf.compat.v1.logging.set_verbosity(old_verbosity)
    except Exception as e:
        # Restore logging verbosity even if an error occurred
        try:
            tf.compat.v1.logging.set_verbosity(old_verbosity)
        except:
            pass
        print(f"Warning: FLOPs calculation failed: {e}")
        metrics["flops"] = 0.0
    
    return metrics


def measure_inference_latency(
    model: keras.Model,
    sample_data: Union[np.ndarray, tf.Tensor, tf.data.Dataset],
    warmup_runs: int = 10,
    test_runs: int = 100,
) -> Dict[str, float]:
    """Measure inference latency with proper warmup.
    
    Args:
        model: Keras model to benchmark
        sample_data: Sample input data
        warmup_runs: Number of warmup iterations
        test_runs: Number of test iterations for measurement
        
    Returns:
        Dictionary with latency statistics
    """
    # Prepare sample input
    if isinstance(sample_data, tf.data.Dataset):
        sample_batch = next(iter(sample_data))
        if isinstance(sample_batch, tuple):
            sample_input = sample_batch[0][:1]
        else:
            sample_input = sample_batch[:1]
    elif isinstance(sample_data, (np.ndarray, tf.Tensor)):
        sample_input = sample_data[:1] if len(sample_data.shape) > 3 else sample_data[np.newaxis, :]
    else:
        raise ValueError("sample_data must be Dataset, ndarray, or Tensor")
    
    # Warmup
    for _ in range(warmup_runs):
        _ = model.predict(sample_input, verbose=0)
    
    # Measure latency
    latencies = []
    for _ in range(test_runs):
        start_time = time.perf_counter()
        _ = model.predict(sample_input, verbose=0)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    
    return {
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "std_latency_ms": float(np.std(latencies)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
    }


def measure_batch_performance(
    model: keras.Model,
    dataset: tf.data.Dataset,
    num_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Measure batch inference performance and throughput.
    
    Args:
        model: Keras model to benchmark
        dataset: Dataset for evaluation
        num_batches: Number of batches to process (None for full dataset)
        
    Returns:
        Dictionary with batch performance metrics
    """
    process = psutil.Process(os.getpid())
    
    # Measure memory before inference
    mem_before = process.memory_info().rss / (1024 ** 2)  # MB
    
    # Warm-up
    sample_batch = next(iter(dataset))
    _ = model.predict(sample_batch[0][:1], verbose=0)
    
    # Measure batch processing time
    start_time = time.perf_counter()
    
    if num_batches:
        batch_count = 0
        total_samples = 0
        for batch in dataset:
            if batch_count >= num_batches:
                break
            inputs = batch[0] if isinstance(batch, tuple) else batch
            _ = model.predict(inputs, verbose=0)
            total_samples += len(inputs)
            batch_count += 1
    else:
        evaluation = model.evaluate(dataset, verbose=0, return_dict=True)
        total_samples = sum(1 for _ in dataset) * dataset.element_spec[0].shape[0]
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Measure memory after inference
    mem_after = process.memory_info().rss / (1024 ** 2)  # MB
    
    return {
        "total_time_s": float(total_time),
        "throughput_fps": float(total_samples / total_time) if total_time > 0 else 0.0,
        "samples_per_second": float(total_samples / total_time) if total_time > 0 else 0.0,
        "memory_used_mb": float(max(mem_after - mem_before, 0)),
        "total_samples": int(total_samples),
    }


def estimate_energy_consumption(
    model: keras.Model,
    platform_config: Dict[str, Any],
    inference_time_s: float,
) -> Dict[str, float]:
    """Estimate energy consumption based on platform characteristics.
    
    Args:
        model: Keras model
        platform_config: Platform specifications (power_budget_w, etc.)
        inference_time_s: Measured inference time in seconds
        
    Returns:
        Dictionary with energy estimates
    """
    # Get model complexity
    metrics = calculate_model_metrics(model)
    
    # Extract platform power characteristics
    power_budget_w = platform_config.get("power_budget_w", 5.0)
    tdp_watts = platform_config.get("tdp_watts", power_budget_w)
    
    # Simple energy model: E = P * t
    # For more accurate models, consider dynamic power based on FLOPs
    base_energy_mj = tdp_watts * inference_time_s
    
    # Dynamic energy based on computational intensity
    # Assume ~1pJ per FLOP (simplified model)
    dynamic_energy_mj = metrics["flops"] * 1e-12 * 1000  # Convert to mJ
    
    total_energy_mj = base_energy_mj + dynamic_energy_mj
    
    return {
        "total_energy_mj": float(total_energy_mj),
        "static_energy_mj": float(base_energy_mj),
        "dynamic_energy_mj": float(dynamic_energy_mj),
        "average_power_w": float(total_energy_mj / 1000 / inference_time_s) if inference_time_s > 0 else 0.0,
        "energy_per_inference_mj": float(total_energy_mj),
    }


def profile_model_comprehensive(
    model: keras.Model,
    test_data: tf.data.Dataset,
    platform_config: Optional[Dict[str, Any]] = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Perform comprehensive model profiling.
    
    Args:
        model: Keras model to profile
        test_data: Test dataset
        platform_config: Platform specifications for energy estimation
        batch_size: Batch size for evaluation
        
    Returns:
        Complete profiling results
    """
    if platform_config is None:
        platform_config = {
            "power_budget_w": 5.0,
            "memory_budget_mb": 1024,
            "tdp_watts": 10.0,
        }
    
    print("Profiling model performance...")
    
    # Basic model metrics
    model_metrics = calculate_model_metrics(model, batch_size=1)
    
    # Latency measurement
    print("  Measuring inference latency...")
    latency_metrics = measure_inference_latency(model, test_data, warmup_runs=5, test_runs=50)
    
    # Batch performance
    print("  Measuring batch performance...")
    batch_metrics = measure_batch_performance(model, test_data, num_batches=10)
    
    # Accuracy evaluation
    print("  Evaluating accuracy...")
    evaluation = model.evaluate(test_data, verbose=0, return_dict=True)
    
    # Energy estimation
    print("  Estimating energy consumption...")
    energy_metrics = estimate_energy_consumption(
        model, platform_config, latency_metrics["mean_latency_ms"] / 1000
    )
    
    # Combine all metrics
    results = {
        **model_metrics,
        **latency_metrics,
        **batch_metrics,
        **energy_metrics,
        "accuracy": float(evaluation.get("accuracy", 0.0)),
        "loss": float(evaluation.get("loss", 0.0)),
    }
    
    return results


def compare_models(
    baseline_results: Dict[str, float],
    optimized_results: Dict[str, float],
    model_name: str = "optimized",
) -> Dict[str, Any]:
    """Compare optimized model against baseline.
    
    Args:
        baseline_results: Profiling results for baseline model
        optimized_results: Profiling results for optimized model
        model_name: Name of the optimized model
        
    Returns:
        Comparison metrics showing improvements/degradations
    """
    comparison = {
        "model_name": model_name,
        "improvements": {},
        "degradations": {},
    }
    
    # Calculate improvements (lower is better)
    for metric in ["mean_latency_ms", "model_size_mb", "total_energy_mj", "memory_used_mb"]:
        if metric in baseline_results and metric in optimized_results:
            baseline_val = baseline_results[metric]
            optimized_val = optimized_results[metric]
            if baseline_val > 0:
                improvement = (baseline_val - optimized_val) / baseline_val * 100
                comparison["improvements"][metric] = float(improvement)
    
    # Calculate throughput improvement (higher is better)
    if "throughput_fps" in baseline_results and "throughput_fps" in optimized_results:
        baseline_fps = baseline_results["throughput_fps"]
        optimized_fps = optimized_results["throughput_fps"]
        if baseline_fps > 0:
            improvement = (optimized_fps - baseline_fps) / baseline_fps * 100
            comparison["improvements"]["throughput_fps"] = float(improvement)
    
    # Calculate accuracy degradation
    if "accuracy" in baseline_results and "accuracy" in optimized_results:
        accuracy_drop = baseline_results["accuracy"] - optimized_results["accuracy"]
        comparison["degradations"]["accuracy_drop"] = float(accuracy_drop)
    
    # Calculate speedup
    if "mean_latency_ms" in baseline_results and "mean_latency_ms" in optimized_results:
        if optimized_results["mean_latency_ms"] > 0:
            speedup = baseline_results["mean_latency_ms"] / optimized_results["mean_latency_ms"]
            comparison["speedup"] = float(speedup)
    
    return comparison


def print_profiling_results(results: Dict[str, Any], title: str = "Model Performance") -> None:
    """Pretty print profiling results.
    
    Args:
        results: Profiling results dictionary
        title: Title for the results section
    """
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}")
    
    # Model characteristics
    print("\nModel Characteristics:")
    print(f"  Parameters: {results.get('parameters', 0):,.0f}")
    print(f"  Model Size: {results.get('model_size_mb', 0):.2f} MB")
    print(f"  FLOPs: {results.get('flops', 0):,.0f}")
    
    # Performance metrics
    print("\nPerformance Metrics:")
    print(f"  Mean Latency: {results.get('mean_latency_ms', 0):.2f} ms")
    print(f"  P95 Latency: {results.get('p95_latency_ms', 0):.2f} ms")
    print(f"  P99 Latency: {results.get('p99_latency_ms', 0):.2f} ms")
    print(f"  Throughput: {results.get('throughput_fps', 0):.2f} FPS")
    
    # Memory metrics
    print("\nMemory Usage:")
    print(f"  Model Memory: {results.get('model_size_mb', 0):.2f} MB")
    print(f"  Runtime Memory: {results.get('memory_used_mb', 0):.2f} MB")
    
    # Energy metrics
    print("\nEnergy Consumption:")
    print(f"  Total Energy: {results.get('total_energy_mj', 0):.2f} mJ")
    print(f"  Average Power: {results.get('average_power_w', 0):.3f} W")
    
    # Accuracy
    print("\nAccuracy:")
    print(f"  Test Accuracy: {results.get('accuracy', 0):.4f} ({results.get('accuracy', 0)*100:.2f}%)")
    print(f"  Test Loss: {results.get('loss', 0):.4f}")
    
    print(f"{'='*70}\n")


def validate_optimization(
    baseline_model: keras.Model,
    optimized_model: keras.Model,
    test_data: tf.data.Dataset,
    platform_config: Optional[Dict[str, Any]] = None,
    accuracy_threshold: float = 0.02,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate that optimization meets requirements.
    
    Args:
        baseline_model: Original baseline model
        optimized_model: Optimized model variant
        test_data: Test dataset
        platform_config: Platform specifications
        accuracy_threshold: Maximum acceptable accuracy drop
        
    Returns:
        Tuple of (validation_passed, detailed_results)
    """
    print("\n" + "="*70)
    print("Validating Optimization".center(70))
    print("="*70)
    
    # Profile both models
    baseline_results = profile_model_comprehensive(baseline_model, test_data, platform_config)
    optimized_results = profile_model_comprehensive(optimized_model, test_data, platform_config)
    
    # Compare results
    comparison = compare_models(baseline_results, optimized_results)
    
    # Validation checks
    checks = {
        "accuracy_preserved": (baseline_results["accuracy"] - optimized_results["accuracy"]) <= accuracy_threshold,
        "latency_improved": optimized_results["mean_latency_ms"] < baseline_results["mean_latency_ms"],
        "memory_reduced": optimized_results["model_size_mb"] < baseline_results["model_size_mb"],
        "energy_reduced": optimized_results["total_energy_mj"] < baseline_results["total_energy_mj"],
    }
    
    validation_passed = checks["accuracy_preserved"] and (
        checks["latency_improved"] or checks["memory_reduced"] or checks["energy_reduced"]
    )
    
    results = {
        "validation_passed": validation_passed,
        "checks": checks,
        "baseline_results": baseline_results,
        "optimized_results": optimized_results,
        "comparison": comparison,
    }
    
    # Print validation summary
    print("\nValidation Results:")
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check_name}: {status}")
    
    print(f"\nOverall: {'✓ VALIDATION PASSED' if validation_passed else '✗ VALIDATION FAILED'}")
    print("="*70 + "\n")
    
    return validation_passed, results


if __name__ == "__main__":
    print("Performance Profiler Module")
    print("This module provides utilities for comprehensive model profiling.")
    print("\nUsage:")
    print("  from performance_profiler import profile_model_comprehensive")
    print("  results = profile_model_comprehensive(model, test_data)")
