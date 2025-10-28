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
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

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
        """Normalise application requirements into a constraint dictionary.

        The returned dictionary captures the hard budgets, soft targets, and
        priority ordering that downstream optimisation components consume.
        """
        defaults = {
            "latency_target_ms": 50.0,
            "throughput_fps": 30.0,
            "power_budget_w": 5.0,
            "memory_budget_mb": 512.0,
            "accuracy_target": 0.85,
            "model_size_mb": 16.0,
        }
        constraints: ConstraintDict = {key.replace("_target", ""): float(application_requirements.get(key, default))
            for key, default in defaults.items()}

        priority_map = application_requirements.get("priority", {})
        constraints["priority_order"] = sorted(
            [key for key in constraints.keys() if key != "priority_order"],
            key=lambda key: priority_map.get(key, 99),
        )

        # Track any requirement that violates the baseline budgets.
        constraints["violations"] = {}
        for key, value in constraints.items():
            if key in ("priority_order", "violations"):
                continue
            min_key = f"min_{key}"
            max_key = f"max_{key}"
            min_bound = application_requirements.get(min_key)
            max_bound = application_requirements.get(max_key)
            if min_bound is not None and value < float(min_bound):
                constraints["violations"][key] = {"type": "below_minimum", "minimum": float(min_bound)}
            if max_bound is not None and value > float(max_bound):
                constraints["violations"][key] = {"type": "above_maximum", "maximum": float(max_bound)}
        return constraints

    def optimization_priority_framework(self, constraints: ConstraintDict) -> List[Dict[str, Any]]:
        """Derive a ranked list of optimisation levers based on constraints."""
        priorities: List[Dict[str, Any]] = []

        def _push(name: str, reason: str, severity: int) -> None:
            priorities.append({"lever": name, "reason": reason, "severity": severity})

        latency_budget = constraints.get("latency_ms", 50.0)
        if latency_budget <= 20:
            _push("latency_optimization", f"Latency budget is tight ({latency_budget} ms)", 1)
        elif latency_budget <= 35:
            _push("latency_optimization", f"Latency budget moderate ({latency_budget} ms)", 2)

        power_budget = constraints.get("power_budget_w", 5.0)
        if power_budget <= 1.5:
            _push("energy_optimization", f"Low power budget ({power_budget} W)", 1)
        elif power_budget <= 3.0:
            _push("energy_optimization", f"Moderate power budget ({power_budget} W)", 2)

        memory_budget = constraints.get("memory_budget_mb", 512.0)
        if memory_budget <= 128.0:
            _push("memory_optimization", f"Memory budget constrained ({memory_budget} MB)", 1)
        elif memory_budget <= 256.0:
            _push("memory_optimization", f"Memory budget moderate ({memory_budget} MB)", 2)

        target_accuracy = constraints.get("accuracy", constraints.get("accuracy_target", 0.85))
        if target_accuracy >= 0.92:
            _push("regularization_and_distillation", f"High accuracy target ({target_accuracy:.2f})", 2)

        if not priorities:
            _push("balanced_multi_objective", "No single constraint dominates", 3)

        priorities.sort(key=lambda item: item["severity"])
        return priorities

    def design_space_exploration(
        self,
        base_model: keras.Model,
        constraints: ConstraintDict,
        alpha_candidates: Optional[List[float]] = None,
        quantization_modes: Optional[List[str]] = None,
        candidate_epochs: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Enumerate candidate optimisation configurations and heuristic scores."""
        alpha_candidates = alpha_candidates or [1.0, 0.75, 0.5, 0.35]
        quantization_modes = quantization_modes or ["float32", "float16", "int8"]
        candidate_epochs = candidate_epochs or [5, 10, 20]

        candidates: List[Dict[str, Any]] = []
        for alpha in alpha_candidates:
            for quantization in quantization_modes:
                for epochs in candidate_epochs:
                    width_penalty = (1.0 - alpha) * 30.0
                    quant_bonus = {"float32": 0.0, "float16": 10.0, "int8": 20.0}.get(quantization, 5.0)
                    energy_score = max(0.0, quant_bonus - width_penalty)
                    latency_headroom = constraints.get("latency_ms", 50.0) - (30.0 * alpha)
                    candidate = {
                        "base_model": base_model.name,
                        "depth_multiplier": alpha,
                        "quantization": quantization,
                        "fine_tune_epochs": epochs,
                        "estimated_energy_score": energy_score,
                        "latency_headroom_ms": latency_headroom,
                        "meets_latency_target": latency_headroom >= 0,
                    }
                    candidates.append(candidate)
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


_DEFAULT_METRIC_CONFIG: Tuple[Tuple[str, str, str, str], ...] = (
    ("Mean Latency (ms)", "mean_latency_ms", "{:.2f}", "min"),
    ("P95 Latency (ms)", "p95_latency_ms", "{:.2f}", "min"),
    ("Throughput (FPS)", "throughput_fps", "{:.2f}", "max"),
    ("Accuracy", "accuracy", "{:.4f}", "max"),
    ("Model Size (MB)", "model_size_mb", "{:.2f}", "min"),
    ("Runtime Memory (MB)", "memory_used_mb", "{:.2f}", "min"),
    ("Total Energy (mJ)", "total_energy_mj", "{:.2f}", "min"),
)

_DEFAULT_PLATFORM_PROFILES: Dict[str, Dict[str, Any]] = {
    "cpu_x86": {
        "peak_compute_gflops": 600.0,
        "memory_bandwidth_gbps": 50.0,
        "tdp_watts": 65.0,
        "l2_cache_kb": 256.0,
    },
    "arm_cortex_a78": {
        "peak_compute_gflops": 200.0,
        "memory_bandwidth_gbps": 15.0,
        "tdp_watts": 5.0,
        "l2_cache_kb": 512.0,
    },
    "arm_cortex_m7": {
        "peak_compute_gflops": 2.0,
        "memory_bandwidth_gbps": 0.5,
        "power_budget_w": 0.1,
        "sram_kb": 512.0,
    },
    "mobile_gpu": {
        "peak_compute_gflops": 900.0,
        "memory_bandwidth_gbps": 34.0,
        "tdp_watts": 6.0,
        "l2_cache_kb": 1024.0,
    },
}


def generate_performance_report(
    profiling_results: Dict[str, Dict[str, float]],
    baseline_model: str = "baseline",
    output_path: Optional[Path] = None,
    metric_config: Optional[Iterable[Tuple[str, str, str, str]]] = None,
) -> Dict[str, Any]:
    """Aggregate profiling metrics into a structured performance report.

    Args:
        profiling_results: Mapping of model name to profiling metrics (as produced by
            ``performance_profiler.profile_model_comprehensive`` or equivalent).
        baseline_model: Key identifying the baseline model in ``profiling_results``.
        output_path: Optional path to persist a Markdown summary table.
        metric_config: Optional iterable overriding the default metric configuration.

    Returns:
        Dictionary containing summary tables, improvement statistics, and Pareto analysis.
    """
    if baseline_model not in profiling_results:
        raise KeyError(
            f"Baseline model '{baseline_model}' not found in profiling_results keys: "
            f"{list(profiling_results.keys())}"
        )

    metric_config = tuple(metric_config or _DEFAULT_METRIC_CONFIG)
    summary_rows: List[Dict[str, Any]] = []
    for model_name, metrics in profiling_results.items():
        row = {"model": model_name}
        for label, key, fmt, _ in metric_config:
            value = metrics.get(key)
            row[label] = None if value is None else float(value)
        summary_rows.append(row)

    baseline_metrics = profiling_results[baseline_model]
    improvements: Dict[str, Dict[str, float]] = {}
    objectives = {key: objective for _, key, _, objective in metric_config}

    for model_name, metrics in profiling_results.items():
        if model_name == baseline_model:
            continue
        model_improvements: Dict[str, float] = {}
        for _, key, _, objective in metric_config:
            baseline_value = baseline_metrics.get(key)
            candidate_value = metrics.get(key)
            if baseline_value in (None, 0) or candidate_value is None:
                continue
            if objective == "min":
                delta = (baseline_value - candidate_value) / baseline_value * 100.0
            else:
                delta = (candidate_value - baseline_value) / baseline_value * 100.0
            model_improvements[key] = float(delta)
        improvements[model_name] = model_improvements

    best_models: Dict[str, str] = {}
    for _, key, _, objective in metric_config:
        comparator = max if objective == "max" else min
        try:
            best_models[key] = comparator(
                profiling_results.items(),
                key=lambda item: float(item[1].get(key, float("inf") if objective == "min" else float("-inf"))),
            )[0]
        except ValueError:
            continue

    def _is_dominated(candidate: Tuple[str, Dict[str, float]], others: List[Tuple[str, Dict[str, float]]]) -> bool:
        name_a, metrics_a = candidate
        for name_b, metrics_b in others:
            if name_a == name_b:
                continue
            better_or_equal = True
            strictly_better = False
            for key, objective in objectives.items():
                a_val = metrics_a.get(key)
                b_val = metrics_b.get(key)
                if a_val is None or b_val is None:
                    better_or_equal = False
                    break
                if objective == "min":
                    if b_val > a_val:
                        better_or_equal = False
                        break
                    if b_val < a_val:
                        strictly_better = True
                else:
                    if b_val < a_val:
                        better_or_equal = False
                        break
                    if b_val > a_val:
                        strictly_better = True
            if better_or_equal and strictly_better:
                return True
        return False

    pareto_frontier = [
        name for name, data in profiling_results.items()
        if not _is_dominated((name, data), list(profiling_results.items()))
    ]

    report = {
        "summary_table": summary_rows,
        "improvements": improvements,
        "best_models": best_models,
        "pareto_frontier": pareto_frontier,
    }

    if output_path is not None:
        output_path = Path(output_path)
        header = ["Model"] + [cfg[0] for cfg in metric_config]
        lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
        for row in summary_rows:
            cells = [row["model"]]
            for _, key, fmt, _ in metric_config:
                value = profiling_results[row["model"]].get(key)
                cells.append("N/A" if value is None else fmt.format(value))
            lines.append("| " + " | ".join(cells) + " |")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")

    return report


def analyze_hardware_utilization(
    profiling_results: Dict[str, Dict[str, float]],
    platform_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Estimate hardware utilisation effectiveness for each model/platform pair.

    Args:
        profiling_results: Mapping of model name to profiling metrics.
        platform_profiles: Optional mapping of platform name to capability metadata.

    Returns:
        Nested dictionary keyed by utilisation category → platform → model → value.
    """
    platform_profiles = platform_profiles or _DEFAULT_PLATFORM_PROFILES
    analysis: Dict[str, Dict[str, Dict[str, float]]] = {
        "cpu_simd_utilization": {},
        "memory_bandwidth_efficiency": {},
        "cache_efficiency": {},
        "thermal_characteristics": {},
        "power_efficiency": {},
    }

    for platform_name, specs in platform_profiles.items():
        peak_gflops = float(specs.get("peak_compute_gflops", specs.get("peak_gflops", 0.0)))
        if peak_gflops == 0.0:
            frequency = float(specs.get("frequency_ghz", 1.0))
            simd_width = float(specs.get("simd_width", 128.0))
            peak_gflops = frequency * simd_width / 32.0 * 2.0  # Heuristic fallback

        memory_bandwidth = float(specs.get("memory_bandwidth_gbps", 0.0))
        cache_capacity_mb = float(specs.get("l2_cache_kb", specs.get("sram_kb", 0.0))) / 1024.0
        tdp_w = float(specs.get("tdp_watts", specs.get("power_budget_w", 0.0)))

        analysis["cpu_simd_utilization"].setdefault(platform_name, {})
        analysis["memory_bandwidth_efficiency"].setdefault(platform_name, {})
        analysis["cache_efficiency"].setdefault(platform_name, {})
        analysis["thermal_characteristics"].setdefault(platform_name, {})
        analysis["power_efficiency"].setdefault(platform_name, {})

        for model_name, metrics in profiling_results.items():
            flops = float(metrics.get("flops", 0.0))
            latency_s = float(metrics.get("mean_latency_ms", 0.0)) / 1000.0
            effective_gflops = 0.0 if latency_s <= 0 else flops / latency_s / 1e9
            simd_util = 0.0 if peak_gflops <= 0 else min(max(effective_gflops / peak_gflops, 0.0), 1.0)

            memory_usage_mb = float(metrics.get("memory_used_mb", metrics.get("model_size_mb", 0.0)))
            mem_throughput_gbps = 0.0
            if latency_s > 0 and memory_usage_mb > 0:
                mem_throughput_gbps = (memory_usage_mb * 8.0) / latency_s / 1000.0
            bandwidth_eff = 0.0 if memory_bandwidth <= 0 else min(max(mem_throughput_gbps / memory_bandwidth, 0.0), 1.0)

            cache_eff = 1.0
            if cache_capacity_mb > 0 and memory_usage_mb > 0:
                cache_eff = min(cache_capacity_mb / memory_usage_mb, 1.0)

            avg_power = float(metrics.get("average_power_w", 0.0))
            if avg_power == 0.0 and latency_s > 0 and metrics.get("total_energy_mj"):
                avg_power = float(metrics["total_energy_mj"]) / 1000.0 / latency_s
            thermal_ratio = 0.0 if tdp_w <= 0 else min(avg_power / tdp_w, 1.0)

            throughput = float(metrics.get("throughput_fps", 0.0))
            power_eff = 0.0
            if avg_power > 0:
                power_eff = throughput / avg_power if throughput > 0 else metrics.get("accuracy", 0.0) / avg_power

            analysis["cpu_simd_utilization"][platform_name][model_name] = float(simd_util)
            analysis["memory_bandwidth_efficiency"][platform_name][model_name] = float(bandwidth_eff)
            analysis["cache_efficiency"][platform_name][model_name] = float(cache_eff)
            analysis["thermal_characteristics"][platform_name][model_name] = float(thermal_ratio)
            analysis["power_efficiency"][platform_name][model_name] = float(power_eff)

    return analysis
