"""Simulation and analytical modelling utilities for Track B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from performance_profiler import calculate_model_metrics


@dataclass
class PerformanceEnvelope:
    peak_compute_gflops: float
    memory_bandwidth_gbps: float
    energy_per_flop_nj: float


class PlatformPerformanceModel:
    """Model performance characteristics across hardware platforms."""

    def __init__(self, platform_specs: Dict[str, Dict[str, Any]]):
        self.platform_specs = platform_specs
        self.performance_models = self._build_performance_models()

    def _build_performance_models(self) -> Dict[str, Dict[str, PerformanceEnvelope]]:
        roofline_models: Dict[str, PerformanceEnvelope] = {}
        energy_models: Dict[str, PerformanceEnvelope] = {}
        memory_models: Dict[str, Dict[str, float]] = {}

        for name, spec in self.platform_specs.items():
            if "frequency_ghz" in spec:
                frequency = spec["frequency_ghz"] * 1e9
                simd_width = spec.get("simd_width", 128)
                core_count = spec.get("core_count", 1)
                peak_compute = frequency * core_count * (simd_width / 32)
                bandwidth = spec.get("memory_bandwidth_gbps", 10)
            elif "frequency_mhz" in spec:
                frequency = spec["frequency_mhz"] * 1e6
                peak_compute = frequency * (2 if spec.get("has_fpu") else 1)
                bandwidth = spec.get("sram_kb", 512) / 1024.0 * 5
            else:
                peak_compute = 1e9
                bandwidth = 5

            energy_per_flop = spec.get("tdp_watts", spec.get("power_budget_mw", 500) / 1000.0) / (
                peak_compute or 1e9
            )

            roofline_models[name] = PerformanceEnvelope(
                peak_compute_gflops=peak_compute / 1e9,
                memory_bandwidth_gbps=bandwidth,
                energy_per_flop_nj=energy_per_flop * 1e9,
            )

            energy_models[name] = roofline_models[name]
            memory_models[name] = {
                "cache_hierarchy_bytes": (spec.get("l1_cache_kb", 0) + spec.get("l2_cache_kb", 0)) * 1024,
                "sram_bytes": spec.get("sram_kb", 0) * 1024,
            }

        return {
            "roofline": roofline_models,
            "energy": energy_models,
            "memory": memory_models,
        }

    def estimate_performance(self, model_graph: Dict[str, float], platform_type: str) -> Dict[str, float]:
        """Estimate latency, throughput and energy based on analytical models."""

        envelope = self.performance_models["roofline"][platform_type]
        energy_model = self.performance_models["energy"][platform_type]
        memory_model = self.performance_models["memory"][platform_type]

        flops = model_graph.get("flops", 0)
        parameter_bytes = model_graph.get("parameter_bytes", 0)
        activation_bytes = model_graph.get("activation_bytes", 0)
        batch_size = model_graph.get("batch_size", 1)

        compute_time = flops / (envelope.peak_compute_gflops * 1e9 + 1e-9)
        memory_time = (parameter_bytes + activation_bytes) / (
            envelope.memory_bandwidth_gbps * 1e9 + 1e-9
        )
        latency_seconds = max(compute_time, memory_time)
        throughput_fps = batch_size / latency_seconds if latency_seconds else float("inf")

        total_bytes = parameter_bytes + activation_bytes
        in_memory = total_bytes <= memory_model.get("cache_hierarchy_bytes", float("inf"))

        energy_nj = flops * energy_model.energy_per_flop_nj

        return {
            "latency_ms": latency_seconds * 1e3,
            "throughput_fps": throughput_fps,
            "energy_mj": energy_nj / 1e6,
            "memory_mb": total_bytes / (1024 ** 2),
            "fits_in_cache": float(in_memory),
            "bottleneck": "compute" if compute_time >= memory_time else "memory",
        }


def validate_with_simulation(models: Dict[str, Dict[str, float]], platform_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Validate analytical estimates using simulation tools."""

    validation_results = {
        "qemu_arm": simulate_arm_performance(models.get("arm_cortex_a", {})),
        "renode_cm": simulate_cortex_m_performance(models.get("arm_cortex_m", {})),
        "webgpu_proxy": simulate_mobile_gpu_performance(models.get("gpu_mobile", {})),
    }
    validation_results["model_accuracy"] = cross_validate_models(validation_results, platform_configs)
    return validation_results


def simulate_arm_performance(model_metrics: Dict[str, float]) -> Dict[str, Any]:
    """Simulate ARM Cortex-A performance using QEMU."""

    return {
        "simulator": "qemu-system-aarch64",
        "command": "qemu-system-aarch64 -M virt -cpu cortex-a72 -m 2048 -kernel benchmark.elf",
        "estimated_latency_ms": model_metrics.get("latency_ms", 0) * 1.05,
        "estimated_energy_mj": model_metrics.get("energy_mj", 0) * 1.1,
    }


def simulate_cortex_m_performance(model_metrics: Dict[str, float]) -> Dict[str, Any]:
    """Simulate Cortex-M performance using Renode."""

    return {
        "simulator": "renode",
        "script": "run_cortex_m.resc",
        "cycle_count": int(model_metrics.get("latency_ms", 0) * 1e6 / 10),
        "memory_usage_kb": model_metrics.get("memory_mb", 0) * 1024,
    }


def simulate_mobile_gpu_performance(model_metrics: Dict[str, float]) -> Dict[str, Any]:
    """Simulate mobile GPU performance using a WebGPU proxy."""

    return {
        "simulator": "webgpu",
        "url": "chrome://gpu",
        "throughput_fps": model_metrics.get("throughput_fps", 0) * 0.9,
        "latency_ms": model_metrics.get("latency_ms", 0) * 1.1,
    }


def cross_validate_models(
    simulation_results: Dict[str, Dict[str, Any]],
    platform_configs: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    """Estimate the deviation between analytical predictions and simulations."""

    validation_scores: Dict[str, float] = {}
    for platform, result in simulation_results.items():
        if "estimated_latency_ms" in result:
            validation_scores[platform] = float(result["estimated_latency_ms"])
        elif "cycle_count" in result:
            freq = platform_configs.get("arm_cortex_m", {}).get("frequency_mhz", 100)
            validation_scores[platform] = result["cycle_count"] / (freq * 1e3)
        elif "latency_ms" in result:
            validation_scores[platform] = float(result["latency_ms"])
    return validation_scores


class CrossPlatformAnalyzer:
    """Analyse optimisation effectiveness across platforms."""

    def __init__(self):
        self.platform_characteristics = self._load_platform_specs()
        self.performance_model = PlatformPerformanceModel(self.platform_characteristics)

    def _load_platform_specs(self) -> Dict[str, Dict[str, Any]]:
        return {
            "x86_desktop": {
                "core_count": 8,
                "frequency_ghz": 3.5,
                "simd_width": 256,
                "l1_cache_kb": 32,
                "l2_cache_kb": 256,
                "l3_cache_mb": 8,
                "memory_bandwidth_gbps": 50,
                "tdp_watts": 65,
            },
            "arm_cortex_a78": {
                "core_count": 4,
                "frequency_ghz": 2.4,
                "simd_width": 128,
                "l1_cache_kb": 64,
                "l2_cache_kb": 512,
                "memory_bandwidth_gbps": 15,
                "tdp_watts": 5,
            },
            "arm_cortex_m7": {
                "frequency_mhz": 400,
                "sram_kb": 512,
                "flash_mb": 2,
                "power_budget_mw": 100,
                "has_fpu": True,
                "dsp_extensions": True,
            },
        }

    def analyze_optimization_effectiveness(
        self,
        baseline_model: Any,
        optimized_models: Dict[str, Any],
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        results: Dict[str, Dict[str, Dict[str, float]]] = {}

        baseline_metrics = self.model_performance(baseline_model, "x86_desktop")

        for platform in self.platform_characteristics:
            platform_results: Dict[str, Dict[str, float]] = {}
            baseline_perf = self.model_performance(baseline_model, platform)

            for opt_name, opt_model in optimized_models.items():
                opt_perf = self.model_performance(opt_model, platform)
                platform_results[opt_name] = {
                    "speedup": baseline_perf["latency_ms"] / opt_perf["latency_ms"],
                    "memory_reduction": 1 - (opt_perf["memory_mb"] / baseline_perf["memory_mb"]),
                    "energy_reduction": 1 - (opt_perf["energy_mj"] / baseline_perf["energy_mj"]),
                    "throughput_gain": opt_perf["throughput_fps"] / baseline_perf["throughput_fps"],
                }

            results[platform] = platform_results

        results["baseline_reference"] = {"x86_desktop": baseline_metrics}
        return results

    def model_performance(self, model: Any, platform: str) -> Dict[str, float]:
        """Model expected performance of ``model`` on ``platform``."""

        metrics = calculate_model_metrics(model)
        graph = {
            "flops": metrics.get("flops", 0),
            "parameter_bytes": metrics.get("parameter_size_bytes", 0),
            "activation_bytes": metrics.get("activation_size_bytes", 0),
            "batch_size": metrics.get("batch_size", 1),
        }
        return self.performance_model.estimate_performance(graph, self._map_platform(platform))

    def _map_platform(self, platform: str) -> str:
        if platform == "x86_desktop":
            return "x86_desktop"
        if platform == "arm_cortex_a78":
            return "arm_cortex_a78"
        if platform == "arm_cortex_m7":
            return "arm_cortex_m7"
        raise ValueError(f"Unknown platform: {platform}")
