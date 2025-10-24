"""Structured optimisation framework for hardware-aware model design."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import tensorflow as tf
from tensorflow import keras

from performance_profiler import calculate_model_metrics


ConstraintDict = Dict[str, Any]


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
        constraints["priority_order"] = sorted(
            ("latency_ms", "throughput_fps", "power_budget_w", "memory_budget_mb", "accuracy_target"),
            key=lambda key: application_requirements.get("priority", {}).get(key, 1),
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
                candidates.append(
                    {
                        "base_model": base_model.name,
                        "depth_multiplier": alpha,
                        "quantization": quantization,
                        "meets_latency_target": quantization != "float32" or constraints["latency_ms"] > 25,
                    }
                )
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
            if isinstance(result, tuple):
                current_model, metadata = result
            else:
                current_model, metadata = result, {}
            history.append({"step": func.__name__, "metadata": metadata})
        return current_model, history


class PerformancePredictor:
    """Predict performance of optimised models on target hardware."""

    def __init__(self, platform_profiles: Dict[str, Dict[str, Any]]):
        self.platform_profiles = platform_profiles

    def predict_latency(self, model: keras.Model, platform: str, batch_size: int = 1) -> float:
        profile = self.platform_profiles[platform]
        metrics = calculate_model_metrics(model, batch_size=batch_size)
        peak_gflops = profile.get("peak_compute_gflops", profile.get("frequency_ghz", 1) * 100)
        return metrics["flops"] / (peak_gflops * 1e9 + 1e-9) * 1e3

    def predict_memory_usage(self, model: keras.Model, platform: str) -> float:
        metrics = calculate_model_metrics(model)
        memory_budget = self.platform_profiles[platform].get("memory_budget_mb", 1024)
        return min(metrics["model_size_mb"], memory_budget)

    def predict_energy_consumption(self, model: keras.Model, platform: str, duration: float = 3600) -> float:
        metrics = calculate_model_metrics(model)
        power_budget = self.platform_profiles[platform].get("power_budget_w", 5.0)
        return power_budget * duration / 1e6 + metrics["flops"] * 1e-12


def create_optimization_report(
    baseline_metrics: Dict[str, float],
    optimized_metrics: Dict[str, float],
    platform: str,
) -> Dict[str, Any]:
    """Create a comparative optimisation report for ``platform``."""

    report = {
        "platform": platform,
        "speedup": baseline_metrics["latency_ms"] / optimized_metrics["latency_ms"],
        "memory_saving": 1 - optimized_metrics["memory_mb"] / baseline_metrics["memory_mb"],
        "energy_reduction": 1 - optimized_metrics["energy_mj"] / baseline_metrics["energy_mj"],
        "throughput_gain": optimized_metrics["throughput_fps"] / baseline_metrics["throughput_fps"],
    }
    return report


def validate_optimization_results(
    model: keras.Model,
    test_data: Any,
    validation_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Validate an optimised model using held-out data."""

    if isinstance(test_data, tf.data.Dataset):
        evaluation = model.evaluate(test_data, verbose=0, return_dict=True)
    elif isinstance(test_data, tuple) and len(test_data) == 2:
        evaluation = model.evaluate(*test_data, verbose=0, return_dict=True)
    else:
        raise TypeError("test_data must be a Dataset or (x, y) tuple")

    validation = {
        "accuracy": evaluation.get("accuracy"),
        "loss": evaluation.get("loss"),
    }
    if validation_metrics:
        validation["regression"] = {
            key: evaluation.get(key, 0) - value for key, value in validation_metrics.items()
        }
    return validation
