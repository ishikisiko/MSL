"""Performance modeling and simulation utilities (Track B).

This module contains analytical performance models and lightweight simulation
placeholders that help validate estimates without access to physical hardware.

CLI usage (wired by Makefile `simulate` target):
        python part3_modeling.py --simulator {qemu,renode,webgpu} \
                [--model-path baseline_mobilenetv2.keras] \
                [--output results/simulation_<sim>.json]

Notes:
- These are QEMU/Renode/WebGPU-style placeholders. They do NOT spin up real
    simulators but provide consistent, reproducible proxies for design-space
    exploration when hardware is unavailable. Integrations can be added later.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from dataclasses import dataclass

from performance_profiler import calculate_model_metrics
import json
import os
import argparse

try:
    # TensorFlow is required only when loading real models for metrics
    import tensorflow as tf
    from tensorflow import keras
except Exception:  # pragma: no cover - allow environments without TF to import module
    tf = None
    keras = None


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


def _default_model(model_path: Optional[str] = None):
    """Load a Keras model if available; otherwise create a tiny fallback.

    This keeps the CLI usable on lightweight environments.
    """
    if model_path and os.path.exists(model_path) and keras is not None:
        try:
            return keras.models.load_model(model_path)
        except Exception:
            pass

    # Fallback: tiny 1-layer model (only used for placeholder metrics)
    if keras is None:
        raise RuntimeError("TensorFlow/Keras not available to build fallback model")
    inputs = keras.Input(shape=(224, 224, 3))
    x = keras.layers.AveragePooling2D(pool_size=(7, 7), strides=7)(inputs)
    x = keras.layers.Conv2D(8, 3, activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model


def _platform_presets(simulator: str) -> Dict[str, Any]:
    """Return reasonable platform spec presets for each simulator type."""
    simulator = simulator.lower()
    if simulator == "qemu":  # ARM Cortex-A class (Linux)
        return {
            "name": "QEMU-ARM-Cortex-A",
            "peak_gflops": 40.0,
            "memory_bandwidth_gbps": 12.0,
            "tdp_watts": 5.0,
            "simulation_overhead_scale": 1.3,
            "energy_scale": 1.1,
        }
    if simulator == "renode":  # ARM Cortex-M class (MCU)
        return {
            "name": "Renode-Cortex-M",
            "peak_gflops": 0.1,  # effectively ~100 MFLOPS class
            "memory_bandwidth_gbps": 0.1,
            "tdp_watts": 0.3,
            "simulation_overhead_scale": 5.0,
        }
    if simulator == "webgpu":  # Mobile GPU proxy
        return {
            "name": "WebGPU-Mobile-Proxy",
            "peak_gflops": 250.0,
            "memory_bandwidth_gbps": 30.0,
            "tdp_watts": 6.0,
            "gpu_scale": 0.45,
            "gpu_energy_scale": 1.0,
        }
    raise ValueError(f"Unsupported simulator: {simulator}")


def _simulate(simulator: str, model, specs: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to simulator-specific placeholder and return results with metadata."""
    ppm = PlatformPerformanceModel(specs)
    analytical = ppm.estimate_performance(model)
    if simulator == "qemu":
        simulated = simulate_arm_performance(model, specs)
    elif simulator == "renode":
        simulated = simulate_cortex_m_performance(model, specs)
    elif simulator == "webgpu":
        simulated = simulate_mobile_gpu_performance(model, specs)
    else:
        raise ValueError(simulator)

    error_score = cross_validate_models(analytical, simulated)
    return {
        "simulator": simulator,
        "platform": specs.get("name", simulator),
        "analytical": analytical,
        "simulated": simulated,
        "cross_validation_error": float(error_score),
    }


def main():  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description="Track B: simulation & modeling")
    parser.add_argument(
        "--simulator",
        required=True,
        choices=["qemu", "renode", "webgpu"],
        help="Simulator placeholder to use",
    )
    parser.add_argument(
        "--model-path",
        default="baseline_mobilenetv2.keras",
        help="Path to a Keras model file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path (default: results/simulation_<sim>.json)",
    )
    args = parser.parse_args()

    # Ensure results directory exists
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = args.output or os.path.join(results_dir, f"simulation_{args.simulator}.json")

    # Load model (or fallback) and run simulation
    model = _default_model(args.model_path)
    specs = _platform_presets(args.simulator)
    result = _simulate(args.simulator, model, specs)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved simulation results → {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
