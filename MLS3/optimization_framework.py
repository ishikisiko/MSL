"""Structured optimisation framework for hardware-aware model design.

REFACTORED (V2):
- Replaced theoretical FLOPs-based latency prediction with a more accurate
  empirical measurement on real data batches.
- Unified the validation logic to efficiently handle both tf.data.Dataset
  and raw (x, y) tuples by applying a preprocessing function.
- Internalized static metric calculation to reduce external dependencies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2_as_graph,
)

ConstraintDict = Dict[str, Any]
AUTOTUNE = tf.data.AUTOTUNE


# --- Internalized Metric Calculation ---

def _calculate_static_metrics(model: keras.Model, batch_size: int = 1) -> Dict[str, float]:
    """Calculates static model metrics like size, parameters, and FLOPs."""
    
    # 1. Model Size
    size_bytes = sum(np.prod(w.shape) * np.dtype(w.dtype.name).itemsize for w in model.weights)
    model_size_mb = size_bytes / (1024 ** 2)

    # 2. FLOPs
    try:
        inputs = tf.TensorSpec([batch_size, *model.input_shape[1:]], tf.float32)
        concrete_fn = tf.function(model).get_concrete_function(inputs)
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_fn)
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name="")
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        total_flops = flops.total_float_ops if flops else 0
    except Exception:
        total_flops = 0  # Fallback if FLOPs calculation fails

    return {
        "model_size_mb": float(model_size_mb),
        "flops": float(total_flops),
        "parameters": float(model.count_params()),
    }


@dataclass
class HardwareAwareDesignMethodology:
    """Framework encapsulating reusable design heuristics."""

    design_principles: List[str] = field(
        default_factory=lambda: [
            "co_design_hardware_software",
            "early_constraint_specification",
            "iterative_optimization",
            "multi_objective_optimization",
            "platform_specific_tuning",
        ]
    )

    def constraint_specification_framework(self, application_requirements: Dict[str, Any]) -> ConstraintDict:
        """Normalise application requirements into a constraint dictionary."""
        constraints = {
            "latency_ms": application_requirements.get("latency_target_ms", 50.0),
            "throughput_fps": application_requirements.get("throughput_fps", 30.0),
            "power_budget_w": application_requirements.get("power_budget_w", 5.0),
            "memory_budget_mb": application_requirements.get("memory_budget_mb", 512),
            "accuracy_target": application_requirements.get("accuracy_target", 0.85),
            "model_size_mb": application_requirements.get("model_size_mb", 16),
        }
        # Sort constraints by user-defined priority
        priority_map = application_requirements.get("priority", {})
        constraints["priority_order"] = sorted(
            constraints.keys(),
            key=lambda key: priority_map.get(key, 99),  # Default to low priority
        )
        return constraints

    def optimization_priority_framework(self, constraints: ConstraintDict) -> List[str]:
        """Derive a ranked list of optimisation levers based on constraints."""
        priorities = []
        if constraints["latency_ms"] < 20:
            priorities.append("latency_optimization")
        if constraints["power_budget_w"] < 2:
            priorities.append("energy_optimization")
        if constraints["memory_budget_mb"] < 128:
            priorities.append("memory_optimization")
        if constraints["accuracy_target"] > 0.9:
            priorities.append("regularization_and_distillation")
        if not priorities:
            priorities.append("balanced_multi_objective")
        return priorities

    def design_space_exploration(
        self,
        base_model: keras.Model,
        constraints: ConstraintDict,
        alpha_candidates: Optional[List[float]] = None,
        quantization_modes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Enumerate candidate optimisation configurations."""
        alpha_candidates = alpha_candidates or [1.0, 0.75, 0.5, 0.35]
        quantization_modes = quantization_modes or ["float32", "float16", "int8"]

        candidates: List[Dict[str, Any]] = []
        for alpha in alpha_candidates:
            for quantization in quantization_modes:
                candidates.append({
                    "base_model": base_model.name,
                    "depth_multiplier": alpha,
                    "quantization": quantization,
                    "meets_latency_target": (quantization != "float32") or (constraints["latency_ms"] > 25),
                })
        return candidates


class OptimizationPipeline:
    """Pipeline that executes optimisation steps with validation hooks."""

    def __init__(self, target_platform: str, constraints: ConstraintDict):
        self.target_platform = target_platform
        self.constraints = constraints
        self.optimization_steps: List[Tuple[Callable[..., Any], Dict[str, Any]]] = []

    def add_optimization_step(self, optimization_func: Callable[..., Any], **kwargs: Any) -> None:
        self.optimization_steps.append((optimization_func, kwargs))

    def execute_pipeline(self, model: keras.Model) -> Tuple[keras.Model, List[Dict[str, Any]]]:
        history: List[Dict[str, Any]] = []
        current_model = model
        for func, kwargs in self.optimization_steps:
            result = func(current_model, **kwargs)
            current_model, metadata = result if isinstance(result, tuple) else (result, {})
            history.append({"step": func.__name__, "metadata": metadata})
        return current_model, history


class PerformancePredictor:
    """Predicts and measures performance of models on target hardware."""

    def __init__(self, platform_profiles: Dict[str, Dict[str, Any]], representative_data: Optional[tf.data.Dataset] = None):
        self.platform_profiles = platform_profiles
        self.representative_data = representative_data

    def _measure_latency_on_real_data(self, model: keras.Model, num_warmup: int = 5, num_timed: int = 20) -> float:
        """Measures single-sample latency using real data."""
        if self.representative_data is None:
            return -1.0  # Indicate that measurement is not possible

        dataset = self.representative_data.unbatch().batch(1)
        
        # Warm-up
        for images, _ in dataset.take(num_warmup):
            _ = model(images, training=False)

        # Measurement
        latencies = []
        for images, _ in dataset.take(num_timed):
            start = time.perf_counter()
            _ = model(images, training=False)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        return float(np.mean(latencies)) if latencies else -1.0

    def predict_latency(self, model: keras.Model, platform: str) -> float:
        """Predicts latency, preferring real measurement over theoretical calculation."""
        # Prefer empirical measurement if data is available
        measured_latency = self._measure_latency_on_real_data(model)
        if measured_latency > 0:
            return measured_latency

        # Fallback to theoretical calculation
        profile = self.platform_profiles.get(platform, {})
        static_metrics = _calculate_static_metrics(model, batch_size=1)
        peak_gflops = profile.get("peak_compute_gflops", profile.get("frequency_ghz", 1.0) * 100)
        theoretical_latency = (static_metrics["flops"] / (peak_gflops * 1e9 + 1e-9)) * 1e3
        return theoretical_latency

    def predict_memory_usage(self, model: keras.Model, platform: str) -> float:
        """Predicts memory usage based on model size and platform budget."""
        static_metrics = _calculate_static_metrics(model)
        memory_budget = self.platform_profiles.get(platform, {}).get("memory_budget_mb", 1024)
        return min(static_metrics["model_size_mb"], memory_budget)

    def predict_energy_consumption(self, model: keras.Model, platform: str, duration_s: float = 3600) -> float:
        """Predicts energy consumption based on platform power budget and model FLOPs."""
        static_metrics = _calculate_static_metrics(model)
        power_budget = self.platform_profiles.get(platform, {}).get("power_budget_w", 5.0)
        # Simplified model: base power + compute energy
        return (power_budget * duration_s / 3600) + (static_metrics["flops"] * 1e-12)


def create_optimization_report(
    baseline_metrics: Dict[str, float],
    optimized_metrics: Dict[str, float],
    platform: str,
) -> Dict[str, Any]:
    """Create a comparative optimisation report for a given platform."""
    report = {"platform": platform}
    
    def _safe_division(num, den):
        return num / den if den != 0 else 0.0

    report["speedup"] = _safe_division(baseline_metrics.get("latency_ms", 0), optimized_metrics.get("latency_ms", 1))
    report["memory_saving_percent"] = (1 - _safe_division(optimized_metrics.get("memory_mb", 0), baseline_metrics.get("memory_mb", 1))) * 100
    report["energy_reduction_percent"] = (1 - _safe_division(optimized_metrics.get("energy_mj", 0), baseline_metrics.get("energy_mj", 1))) * 100
    report["throughput_gain_percent"] = (_safe_division(optimized_metrics.get("throughput_fps", 0), baseline_metrics.get("throughput_fps", 1)) - 1) * 100
    
    return report


def validate_optimization_results(
    model: keras.Model,
    test_data: Any,
    baseline_metrics: Optional[Dict[str, float]] = None,
    preprocess_func: Optional[Callable] = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Validate an optimised model using held-out data, with efficient preprocessing."""
    
    if isinstance(test_data, tuple) and len(test_data) == 2 and preprocess_func is not None:
        # Efficiently create a dataset from raw data using the provided preprocessor
        x, y = test_data
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        ds = ds.map(preprocess_func, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    elif isinstance(test_data, tf.data.Dataset):
        ds = test_data  # Assume dataset is already batched and prefetched
    else:
        raise TypeError("test_data must be a tf.data.Dataset, or an (x, y) tuple with a valid preprocess_func.")

    evaluation = model.evaluate(ds, verbose=0, return_dict=True)

    validation = {
        "accuracy": evaluation.get("accuracy"),
        "loss": evaluation.get("loss"),
    }
    if baseline_metrics:
        validation["regression"] = {
            key: evaluation.get(key, 0) - baseline_metrics.get(key, 0)
            for key in ["accuracy", "loss"]
        }
    return validation
