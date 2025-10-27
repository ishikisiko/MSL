"""Performance modeling and simulation utilities (Track B).

This module contains analytical performance models and lightweight simulation
placeholders that help validate estimates without access to physical hardware.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from dataclasses import dataclass

from performance_profiler import calculate_model_metrics


@dataclass
class PlatformPerformanceModel:
    platform_specs: Dict[str, Any]

    def _build_roofline_model(self) -> Dict[str, Any]:
        # Very small roofline model derived from specs
        peak_gflops = self.platform_specs.get("peak_gflops", self.platform_specs.get("frequency_ghz", 1) * 100)
        mem_bandwidth_gbps = self.platform_specs.get("memory_bandwidth_gbps", 10)
        return {"peak_gflops": peak_gflops, "mem_bandwidth_gbps": mem_bandwidth_gbps}

    def _build_energy_model(self) -> Dict[str, Any]:
        # Simple energy model using TDP and efficiency scalar
        tdp = self.platform_specs.get("tdp_watts", self.platform_specs.get("power_budget_w", 5.0))
        eff = self.platform_specs.get("compute_efficiency_ops_per_joule", 1e9)
        return {"tdp_watts": tdp, "efficiency_ops_per_joule": eff}

    def estimate_performance(self, model, platform_type: Optional[str] = None) -> Dict[str, float]:
        """Estimate latency, memory and energy for a given Keras model.

        This function is intentionally lightweight and uses measured FLOPs from
        the `performance_profiler` to produce relative estimates suitable for
        design-space exploration.
        """
        metrics = calculate_model_metrics(model, batch_size=1)
        roof = self._build_roofline_model()
        energy = self._build_energy_model()

        # Estimate latency (ms)
        peak_ops_per_s = roof["peak_gflops"] * 1e9
        flops = metrics.get("flops", 0.0)
        est_latency_s = flops / (peak_ops_per_s + 1e-12)

        # Estimate memory usage
        mem_mb = metrics.get("model_size_mb", 0.0)

        # Estimate energy (mJ)
        est_energy_j = flops / energy["efficiency_ops_per_joule"]
        est_energy_mj = est_energy_j * 1000

        return {
            "latency_ms": float(est_latency_s * 1000),
            "memory_mb": float(mem_mb),
            "energy_mj": float(est_energy_mj),
            "flops": float(flops),
            "parameters": float(metrics.get("parameters", 0.0)),
        }


def simulate_arm_performance(model, platform_specs: Dict[str, Any]) -> Dict[str, float]:
    """Lightweight QEMU-style simulation placeholder.

    Returns synthetic measured metrics by scaling the analytical estimates.
    """
    ppm = PlatformPerformanceModel(platform_specs)
    est = ppm.estimate_performance(model)
    # Apply empirical scale factors to emulate simulation overhead
    est["latency_ms"] *= platform_specs.get("simulation_overhead_scale", 2.0)
    est["energy_mj"] *= platform_specs.get("energy_scale", 1.2)
    return est


def simulate_cortex_m_performance(model, platform_specs: Dict[str, Any]) -> Dict[str, float]:
    ppm = PlatformPerformanceModel(platform_specs)
    est = ppm.estimate_performance(model)
    est["latency_ms"] *= platform_specs.get("simulation_overhead_scale", 5.0)
    est["memory_mb"] *= 0.5  # MCUs often have very limited memory; report model fit ratio
    return est


def simulate_mobile_gpu_performance(model, platform_specs: Dict[str, Any]) -> Dict[str, float]:
    ppm = PlatformPerformanceModel(platform_specs)
    est = ppm.estimate_performance(model)
    est["latency_ms"] *= platform_specs.get("gpu_scale", 0.5)
    est["energy_mj"] *= platform_specs.get("gpu_energy_scale", 1.0)
    return est


def cross_validate_models(analytical: Dict[str, float], simulated: Dict[str, float]) -> float:
    """Return a simple accuracy score between analytical and simulated estimates."""
    keys = ["latency_ms", "memory_mb", "energy_mj"]
    diffs = []
    for k in keys:
        a = analytical.get(k, 0.0)
        s = simulated.get(k, 0.0)
        if a + s == 0:
            diffs.append(0.0)
        else:
            diffs.append(abs(a - s) / max(a, s, 1e-12))
    return float(np.mean(diffs))


__all__ = [
    "PlatformPerformanceModel",
    "simulate_arm_performance",
    "simulate_cortex_m_performance",
    "simulate_mobile_gpu_performance",
    "cross_validate_models",
]
