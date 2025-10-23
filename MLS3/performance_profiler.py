"""Utilities for profiling model performance across hardware targets."""

from __future__ import annotations

import os
import statistics
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import psutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2_as_graph,
)


def generate_performance_report(
    model_metrics: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Generate a consolidated performance report for all models/platforms."""

    report = {
        "models": list(model_metrics.keys()) if model_metrics else [],
        "platforms": [
            "cpu_x86",
            "arm_cortex_a",
            "arm_cortex_m",
            "mobile_gpu",
        ],
        "metrics": [
            "latency_ms",
            "memory_mb",
            "energy_mj",
            "accuracy",
            "throughput_fps",
        ],
        "entries": model_metrics or {},
    }
    return report


def analyze_hardware_utilization(
    latency_results: Dict[str, Dict[str, float]],
    platform_specs: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Estimate hardware utilisation based on latency and platform specs."""

    utilization = {
        "cpu_simd_utilization": {},
        "memory_bandwidth_efficiency": {},
        "cache_efficiency": {},
        "thermal_characteristics": {},
        "power_efficiency": {},
    }

    for platform, metrics in latency_results.items():
        spec = platform_specs.get(platform, {})
        latency = metrics.get("latency_ms", 0)
        throughput = metrics.get("throughput_fps", 0)
        energy = metrics.get("energy_mj", 0)

        simd_width = spec.get("simd_width", 128)
        peak_fps = spec.get("core_count", 1) * spec.get("frequency_ghz", 1) * 100
        utilization["cpu_simd_utilization"][platform] = min(throughput / (peak_fps + 1e-6), 1.0)

        bandwidth = spec.get("memory_bandwidth_gbps", 10)
        data_moved_gb = metrics.get("memory_mb", 0) / 1024
        time_seconds = latency / 1e3
        effective_bandwidth = data_moved_gb / (time_seconds + 1e-9)
        utilization["memory_bandwidth_efficiency"][platform] = min(effective_bandwidth / (bandwidth + 1e-9), 1.0)

        cache_size_mb = (spec.get("l1_cache_kb", 0) + spec.get("l2_cache_kb", 0)) / 1024
        utilization["cache_efficiency"][platform] = min(metrics.get("memory_mb", 0) / (cache_size_mb + 1e-6), 1.0)

        tdp = spec.get("tdp_watts", spec.get("power_budget_mw", 1000) / 1000.0)
        utilization["power_efficiency"][platform] = min((energy / (time_seconds + 1e-9)) / (tdp + 1e-6), 1.0)
        utilization["thermal_characteristics"][platform] = float(tdp > 10)

    return utilization


def _prepare_dataset(dataset: Any, batch_size: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    if isinstance(dataset, tf.data.Dataset):
        return dataset
    if isinstance(dataset, tuple) and len(dataset) == 2:
        x, y = dataset
        return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
    raise TypeError("dataset must be a tf.data.Dataset or (x, y) tuple")


def benchmark_model_latency(
    model: keras.Model,
    dataset: Any,
    num_runs: int = 100,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Benchmark model latency with warm-up and percentile statistics."""

    ds = _prepare_dataset(dataset, batch_size).cache().prefetch(tf.data.AUTOTUNE)
    iterator = iter(ds.repeat())

    # Warm-up
    for _ in range(min(5, num_runs // 5 + 1)):
        batch = next(iterator)
        model.predict(batch[0], verbose=0)

    latencies: List[float] = []
    for _ in range(num_runs):
        batch = next(iterator)
        start = time.perf_counter()
        model.predict(batch[0], verbose=0)
        end = time.perf_counter()
        latencies.append((end - start) * 1e3)

    latencies.sort()
    return {
        "mean_ms": statistics.fmean(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": latencies[int(len(latencies) * 0.95) - 1],
        "min_ms": latencies[0],
        "max_ms": latencies[-1],
    }


def measure_memory_usage(
    model: keras.Model,
    dataset: Any,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Measure memory consumption during inference."""

    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss

    ds = _prepare_dataset(dataset, batch_size).take(5)
    peak_mem = start_mem
    for batch in ds:
        model.predict(batch[0], verbose=0)
        peak_mem = max(peak_mem, process.memory_info().rss)

    end_mem = process.memory_info().rss
    return {
        "baseline_mb": start_mem / (1024 ** 2),
        "peak_mb": peak_mem / (1024 ** 2),
        "delta_mb": (peak_mem - start_mem) / (1024 ** 2),
        "residual_mb": end_mem / (1024 ** 2),
    }


def estimate_energy_consumption(
    model: keras.Model,
    dataset: Any,
    duration_seconds: float = 60.0,
    power_budget_watts: float = 5.0,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Estimate the energy consumption for running inference on ``dataset``."""

    ds = _prepare_dataset(dataset, batch_size).repeat()
    iterator = iter(ds)

    start_time = time.perf_counter()
    inferences = 0
    while time.perf_counter() - start_time < duration_seconds:
        batch = next(iterator)
        model.predict(batch[0], verbose=0)
        inferences += len(batch[0])

    energy_joules = power_budget_watts * duration_seconds
    return {
        "estimated_energy_mj": energy_joules / 1e6,
        "throughput_fps": inferences / duration_seconds,
        "power_budget_watts": power_budget_watts,
    }


def calculate_model_metrics(
    model: keras.Model,
    batch_size: int = 1,
) -> Dict[str, float]:
    """Calculate FLOPs, parameter count and activation footprint for ``model``."""

    if not model.built:
        raise ValueError("Model must be built before profiling.")

    param_count = model.count_params()
    parameter_size_bytes = sum(np.prod(w.shape) * w.dtype.size for w in model.weights)

    activation_size_bytes = 0
    for layer in model.layers:
        if not hasattr(layer, "output_shape"):
            continue
        output_shape = layer.output_shape
        if isinstance(output_shape, list):
            for shape in output_shape:
                activation_size_bytes += _tensor_size_bytes(shape, layer.dtype)
        else:
            activation_size_bytes += _tensor_size_bytes(output_shape, layer.dtype)

    flops = _compute_model_flops(model, batch_size)

    return {
        "flops": float(flops),
        "parameter_count": float(param_count),
        "parameter_size_bytes": float(parameter_size_bytes),
        "activation_size_bytes": float(activation_size_bytes),
        "model_size_mb": float(parameter_size_bytes / (1024 ** 2)),
        "batch_size": batch_size,
    }


def _tensor_size_bytes(shape: Sequence[int], dtype: tf.dtypes.DType) -> int:
    if not shape or None in shape:
        return 0
    dtype = tf.as_dtype(dtype or tf.float32)
    return int(np.prod(shape) * dtype.size)


def _compute_model_flops(model: keras.Model, batch_size: int) -> int:
    try:
        inputs = tf.TensorSpec([batch_size, *model.input_shape[1:]], tf.float32)
        concrete_fn = tf.function(model).get_concrete_function(inputs)
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_fn)
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name="")
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops if flops is not None else 0
    except Exception:  # pragma: no cover - defensive fallback.
        return 0
