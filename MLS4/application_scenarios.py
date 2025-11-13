from __future__ import annotations

from typing import Any, Dict, List, Optional


class ApplicationScenarioAnalysis:
    """
    Translate raw benchmarking metrics into deployment-specific guidance.

    Args:
        compressed_models: Iterable of evaluation dictionaries produced by
            `CompressionEvaluator.benchmark_compressed_model`.
    """

    def __init__(self, compressed_models: List[Dict[str, Any]]):
        self.compressed_models = compressed_models

    def mobile_deployment_optimization(self) -> Dict[str, Any]:
        """
        Optimize compression for mobile deployment constraints.
        """

        if not self.compressed_models:
            return {}

        prioritized = sorted(
            self.compressed_models,
            key=lambda item: (
                item.get("single_inference_ms", float("inf")),
                -item.get("compression_ratio", 0.0),
            ),
        )
        best = prioritized[0]
        battery_impact = self._estimate_battery_impact(best)
        thermal = self._estimate_thermal_budget(best)

        return {
            "optimal_compression_pipeline": best,
            "battery_life_impact": battery_impact,
            "thermal_considerations": thermal,
            "user_experience_metrics": {
                "latency_ms": best.get("single_inference_ms"),
                "model_size_mb": best.get("model_size_mb"),
            },
        }

    def edge_device_optimization(self) -> Dict[str, Any]:
        """
        Optimize compression for edge computing scenarios.
        """

        if not self.compressed_models:
            return {}

        prioritized = sorted(
            self.compressed_models,
            key=lambda item: (
                item.get("peak_memory_mb", float("inf")),
                item.get("batch_inference_ms", float("inf")),
            ),
        )
        best = prioritized[0]
        memory_budget = max(1.0, best.get("peak_memory_mb", 1.0))

        return {
            "memory_constrained_optimization": best,
            "real_time_performance": {
                "batch_latency_ms": best.get("batch_inference_ms"),
                "throughput_estimate_fps": 1000.0
                / max(1.0, best.get("batch_inference_ms", 1.0)),
            },
            "device_scalability": {
                "fits_microcontroller": memory_budget < 32,
                "fits_edge_gpu": memory_budget < 256,
            },
            "deployment_considerations": {
                "requires_quantization": best.get("compression_ratio", 1.0) > 4.0,
                "suggested_batch_size": 1 if memory_budget < 64 else 4,
            },
        }

    def cloud_inference_optimization(self) -> Dict[str, Any]:
        """
        Optimize compression for cloud-based inference.
        """

        if not self.compressed_models:
            return {}

        prioritized = sorted(
            self.compressed_models,
            key=lambda item: (
                -item.get("test_accuracy", 0.0),
                item.get("batch_inference_ms", float("inf")),
            ),
        )
        best = prioritized[0]
        throughput = 1000.0 / max(1.0, best.get("batch_inference_ms", 1.0))

        return {
            "throughput_optimization": {
                "model": best,
                "throughput_fps": throughput,
            },
            "batch_processing_efficiency": {
                "optimal_batch": 16 if throughput > 200 else 8,
                "autoscaling_hint": "scale_up" if best.get("test_accuracy", 0.0) > 0.75 else "scale_out",
            },
            "cost_benefit_analysis": {
                "compression_ratio": best.get("compression_ratio"),
                "energy_consumption": best.get("energy_consumption"),
            },
            "serving_infrastructure_impact": {
                "gpu_friendly": best.get("flops", 0.0) > 1e9,
                "cpu_only_feasible": throughput > 100,
            },
        }

    # ------------------------------------------------------------------ #
    # Internal heuristics
    # ------------------------------------------------------------------ #
    def _estimate_battery_impact(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        latency = metrics.get("single_inference_ms", 0.0)
        compression = metrics.get("compression_ratio", 1.0)
        impact = max(0.1, min(1.0, latency / 50.0)) * (1 / compression)
        return {
            "relative_draw": impact,
            "session_minutes_estimate": 60.0 / impact if impact else 60.0,
        }

    def _estimate_thermal_budget(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        latency = metrics.get("single_inference_ms", 0.0)
        energy = metrics.get("energy_consumption", 0.0)
        thermal_score = energy / max(1.0, latency)
        return {
            "thermal_score": thermal_score,
            "requires_heatsink": thermal_score > 5.0,
        }
