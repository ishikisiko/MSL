from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tf_compat  # noqa: F401  # maintain tfmot compatibility before TensorFlow import
import tensorflow as tf


@dataclass
class MetricAccumulator:
    accuracy: List[float] = field(default_factory=list)
    model_size_mb: List[float] = field(default_factory=list)
    inference_latency_ms: List[float] = field(default_factory=list)
    memory_usage_mb: List[float] = field(default_factory=list)
    flops_reduction: List[float] = field(default_factory=list)
    compression_ratio: List[float] = field(default_factory=list)
    energy_consumption: List[float] = field(default_factory=list)


class CompressionEvaluator:
    """
    Utility responsible for benchmarking compressed models across metrics.
    """

    def __init__(self, baseline_metrics: Optional[Dict[str, float]] = None) -> None:
        self.metrics = MetricAccumulator()
        self.baseline_metrics = baseline_metrics or {}

    def set_baseline_metrics(self, baseline_metrics: Dict[str, float]) -> None:
        self.baseline_metrics = baseline_metrics

    def benchmark_compressed_model(
        self,
        model: tf.keras.Model,
        test_data: tf.data.Dataset,
        model_name: str,
        technique: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Comprehensive benchmarking of compressed models.
        """

        accuracy = float(model.evaluate(test_data, verbose=0)[1])  # type: ignore[index]
        model_size_mb = self._estimate_model_size(model)
        latency_single, latency_batch = self._measure_latency(model, test_data)
        peak_memory = self._estimate_activation_memory(model, test_data)
        flops = self._estimate_flops(model)
        compression_ratio = self._compute_compression_ratio(model_size_mb)
        baseline_accuracy = self.baseline_metrics.get("test_accuracy")
        accuracy_loss = (
            max(0.0, baseline_accuracy - accuracy) / baseline_accuracy * 100
            if baseline_accuracy
            else 0.0
        )
        energy_estimate = latency_batch * 0.5  # proxy coefficient

        metrics = {
            "model_name": model_name,
            "technique": technique,
            "test_accuracy": accuracy,
            "model_size_mb": model_size_mb,
            "single_inference_ms": latency_single,
            "batch_inference_ms": latency_batch,
            "peak_memory_mb": peak_memory,
            "flops": flops,
            "compression_ratio": compression_ratio,
            "accuracy_loss_percent": accuracy_loss,
            "energy_consumption": energy_estimate,
        }

        self.metrics.accuracy.append(accuracy)
        self.metrics.model_size_mb.append(model_size_mb)
        self.metrics.inference_latency_ms.append(latency_batch)
        self.metrics.memory_usage_mb.append(peak_memory)
        self.metrics.flops_reduction.append(flops)
        self.metrics.compression_ratio.append(compression_ratio)
        self.metrics.energy_consumption.append(energy_estimate)

        return metrics

    def cross_validate_compression(
        self,
        compression_function: Callable[[int], Dict[str, float]],
        seeds: Sequence[int] = (42, 1337, 2024),
    ) -> Dict[str, Any]:
        """
        Validate compression results across multiple random seeds.
        """

        records = []
        for seed in seeds:
            record = compression_function(seed)
            records.append(record)

        mean_accuracy = statistics.fmean(record["accuracy"] for record in records)
        std_accuracy = statistics.pstdev(record["accuracy"] for record in records)

        return {
            "records": records,
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "seeds": list(seeds),
        }

    def analyze_compression_robustness(
        self,
        compressed_model: tf.keras.Model,
        test_data: tf.data.Dataset,
        corruption_std: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Evaluate robustness of compressed models under noise/corruption.
        """

        clean_accuracy = float(compressed_model.evaluate(test_data, verbose=0)[1])  # type: ignore[index]

        corrupted = test_data.map(
            lambda x, y: (
                tf.clip_by_value(
                    x + tf.random.normal(tf.shape(x), stddev=corruption_std),
                    -3.0,
                    3.0,
                ),
                y,
            )
        )
        corrupted_accuracy = float(
            compressed_model.evaluate(corrupted, verbose=0)[1]
        )  # type: ignore[index]

        return {
            "clean_accuracy": clean_accuracy,
            "corrupted_accuracy": corrupted_accuracy,
            "accuracy_drop": clean_accuracy - corrupted_accuracy,
            "corruption_std": corruption_std,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _estimate_model_size(self, model: tf.keras.Model) -> float:
        total_bytes = 0
        for weight in model.weights:
            tensor = weight.numpy()
            total_bytes += tensor.nbytes
        return total_bytes / (1024 * 1024)

    def _measure_latency(
        self,
        model: tf.keras.Model,
        test_data: tf.data.Dataset,
    ) -> Tuple[float, float]:
        iterator = iter(test_data)
        sample_batch = next(iterator)
        inputs = sample_batch[0]
        warm_start = time.perf_counter()
        _ = model(inputs, training=False)
        _ = model(inputs, training=False)
        warm_duration = time.perf_counter() - warm_start

        start = time.perf_counter()
        _ = model(inputs, training=False)
        single_duration = (time.perf_counter() - start) * 1000

        batched_inputs = tf.concat([inputs, inputs], axis=0)
        start_batch = time.perf_counter()
        _ = model(batched_inputs, training=False)
        batch_duration = (time.perf_counter() - start_batch) * 1000
        _ = warm_duration  # suppress lint
        return single_duration, batch_duration

    def _estimate_activation_memory(
        self,
        model: tf.keras.Model,
        test_data: tf.data.Dataset,
    ) -> float:
        sample = next(iter(test_data))[0]
        activation_sizes = []
        for layer in model.layers:
            output_shape = getattr(layer, "output_shape", None)
            if output_shape is None and hasattr(layer, "batch_input_shape"):
                output_shape = layer.batch_input_shape
            if output_shape is None and hasattr(layer, "compute_output_shape"):
                try:
                    output_shape = layer.compute_output_shape(sample.shape)
                except Exception:  # pragma: no cover - defensive fallback for custom layers
                    output_shape = None
            if output_shape is None:
                continue
            if isinstance(output_shape, list):
                output_shape = output_shape[0]
            if output_shape is None or None in output_shape:
                continue
            size = np.prod(output_shape) * 4 / (1024 * 1024)
            activation_sizes.append(float(size))
        return max(activation_sizes) if activation_sizes else float(sample.numpy().nbytes / (1024 * 1024))

    def _estimate_flops(self, model: tf.keras.Model) -> float:
        total_flops = 0
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                kernel_h, kernel_w = layer.kernel_size
                kernel = getattr(layer, "kernel", None)
                in_channels = kernel.shape[2] if kernel is not None else None
                if in_channels is None:
                    input_tensor = getattr(layer, "input", None)
                    input_shape = getattr(input_tensor, "shape", None)
                    if isinstance(input_shape, tf.TensorShape):
                        input_shape = input_shape.as_list()
                    if isinstance(input_shape, (tuple, list)) and input_shape:
                        in_channels = input_shape[-1]
                if hasattr(in_channels, "value"):
                    in_channels = in_channels.value
                if in_channels is None:
                    continue
                out_channels = layer.filters
                output_shape = getattr(layer, "output_shape", None)
                if isinstance(output_shape, tf.TensorShape):
                    output_shape = output_shape.as_list()
                if output_shape is None:
                    tensor = getattr(layer, "output", None)
                    tensor_shape = getattr(tensor, "shape", None)
                    if isinstance(tensor_shape, tf.TensorShape):
                        output_shape = tensor_shape.as_list()
                    else:
                        output_shape = tensor_shape
                if not isinstance(output_shape, (tuple, list)) or len(output_shape) < 3:
                    continue
                output_h, output_w = output_shape[1], output_shape[2]
                if hasattr(output_h, "value"):
                    output_h = output_h.value
                if hasattr(output_w, "value"):
                    output_w = output_w.value
                if None in (kernel_h, kernel_w, in_channels, out_channels, output_h, output_w):
                    continue
                total_flops += 2 * int(kernel_h) * int(kernel_w) * int(in_channels) * int(out_channels) * int(output_h) * int(output_w)
            elif isinstance(layer, tf.keras.layers.Dense):
                if layer.kernel is not None:
                    input_units = layer.kernel.shape[0]
                else:
                    previous = getattr(layer, "input", None)
                    input_shape = getattr(previous, "shape", None)
                    if isinstance(input_shape, tf.TensorShape):
                        input_shape = input_shape.as_list()
                    input_units = input_shape[-1] if isinstance(input_shape, (tuple, list)) and input_shape else None
                if hasattr(input_units, "value"):
                    input_units = input_units.value
                if input_units is None:
                    continue
                total_flops += 2 * int(input_units) * int(layer.units)
        return float(total_flops)

    def _compute_compression_ratio(self, model_size_mb: float) -> float:
        baseline_size = self.baseline_metrics.get("model_size_mb", model_size_mb)
        return baseline_size / model_size_mb if model_size_mb else 1.0
