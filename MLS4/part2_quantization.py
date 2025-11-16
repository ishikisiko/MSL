from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import tf_compat  # noqa: F401  # enforce legacy tf.keras mode for compatibility
import tensorflow as tf

from baseline_model import CUSTOM_OBJECTS, prepare_compression_datasets


AUTOTUNE = tf.data.AUTOTUNE


@dataclass
class DatasetBundle:
    train: tf.data.Dataset
    val: tf.data.Dataset
    test: tf.data.Dataset
    calibration: Tuple[np.ndarray, np.ndarray]
    train_size: int
    val_size: int
    test_size: int


class QuantizationPipeline:
    """
    Implements the quantization experiments described in Part II.

    The class relies on lightweight fake-quantization to keep experiments fast
    and differentiable, while remaining faithful to the behaviours we care
    about (layer sensitivity, PTQ vs QAT, extreme precision regimes, etc.).
    """

    def __init__(
        self,
        model: tf.keras.Model,
        cache_datasets: bool = True,
        default_batch_size: int = 128,
    ) -> None:
        self.base_model = model
        self.quantization_results: Dict[str, Dict] = {}
        self.cache_datasets = cache_datasets
        self._dataset_bundle: Optional[DatasetBundle] = None
        self._cached_batch_size = default_batch_size
        self._optimizer_config = tf.keras.optimizers.serialize(
            model.optimizer
            if getattr(model, "optimizer", None)
            else tf.keras.optimizers.Adam(learning_rate=1e-3)
        )
        loss_obj = (
            model.loss
            if getattr(model, "loss", None)
            else tf.keras.losses.SparseCategoricalCrossentropy()
        )
        self._loss_config = tf.keras.losses.serialize(loss_obj)
        if getattr(model, "metrics", None):
            self._metric_configs = [
                tf.keras.metrics.serialize(metric) for metric in model.metrics
            ]
        else:
            self._metric_configs = [
                tf.keras.metrics.serialize(
                    tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
                )
            ]

        if cache_datasets:
            self._dataset_bundle = self._prepare_datasets(batch_size=default_batch_size)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def mixed_bit_quantization(
        self,
        bit_configurations: List[int] = [8, 4, 2],
        batch_size: int = 128,
    ) -> Dict:
        """
        Assign heterogeneous bit-widths across layers using sensitivity analysis.
        """

        if not bit_configurations:
            raise ValueError("bit_configurations must include at least one entry.")

        quant_layers = self._quantizable_layers(self.base_model)
        if not quant_layers:
            raise RuntimeError("Model contains no quantizable layers.")

        sensitivity = {
            layer.name: self._estimate_layer_sensitivity(layer) for layer in quant_layers
        }
        assignment = self._assign_bits(bit_configurations, sensitivity)

        quant_model = self._quantize_model(bit_assignment=assignment)
        val_ds = self._get_dataset("val", batch_size=batch_size)
        accuracy = self._evaluate_model(quant_model, val_ds)
        compression_ratio = self._estimate_compression_ratio(assignment)

        mixed_results = {
            "model": quant_model,
            "accuracy": accuracy,
            "compression_ratio": compression_ratio,
            "bit_assignment": assignment,
        }
        analysis = {
            "sensitivity_analysis": sensitivity,
            "optimal_bit_assignment": assignment,
            "mixed_bit_models": {"adaptive_assignment": mixed_results},
            "compression_analysis": {
                "baseline_bits": 32.0,
                "effective_bits": np.mean(list(assignment.values())),
                "compression_ratio": compression_ratio,
            },
        }

        self.quantization_results["mixed_bit"] = analysis
        return analysis

    def post_training_vs_qat_comparison(
        self,
        bit_widths: List[int] = [8, 4],
        qat_epochs: int = 2,
        qat_steps: int = 40,
        batch_size: int = 128,
    ) -> Dict:
        """
        Compare post-training quantization against simple quantization-aware tuning.
        """

        val_ds = self._get_dataset("val", batch_size=batch_size)
        train_subset = (
            self._get_dataset("train", batch_size=batch_size).take(qat_steps).repeat()
        )

        comparison = {
            "ptq_results": {},
            "qat_results": {},
            "accuracy_comparison": {},
            "training_efficiency": {},
            "calibration_analysis": {"dataset_examples": self._calibration_size()},
        }

        for bits in bit_widths:
            ptq_model = self._quantize_model(default_bits=bits)
            ptq_accuracy = self._evaluate_model(ptq_model, val_ds)
            comparison["ptq_results"][bits] = {
                "model": ptq_model,
                "accuracy": ptq_accuracy,
                "compression_ratio": self._estimate_compression_ratio(
                    default_bits=bits
                ),
            }

            qat_model = self._quantize_model(default_bits=bits)
            start = time.perf_counter()
            history = self._quantization_aware_finetune(
                qat_model,
                train_subset=train_subset,
                epochs=qat_epochs,
                steps_per_epoch=qat_steps,
                bits=bits,
            )
            duration = time.perf_counter() - start
            qat_accuracy = self._evaluate_model(qat_model, val_ds)
            comparison["qat_results"][bits] = {
                "model": qat_model,
                "accuracy": qat_accuracy,
                "history": history,
                "compression_ratio": self._estimate_compression_ratio(
                    default_bits=bits
                ),
            }
            comparison["accuracy_comparison"][bits] = {
                "ptq": ptq_accuracy,
                "qat": qat_accuracy,
                "gain": qat_accuracy - ptq_accuracy,
            }
            comparison["training_efficiency"][bits] = {
                "epochs": qat_epochs,
                "steps": qat_steps,
                "duration_sec": duration,
            }

        self.quantization_results["ptq_vs_qat"] = comparison
        return comparison

    def extreme_quantization(
        self,
        batch_size: int = 128,
    ) -> Dict:
        """
        Explore INT4 and binary quantization regimes.
        """

        val_ds = self._get_dataset("val", batch_size=batch_size)
        int4_model = self._quantize_model(default_bits=4)
        int4_accuracy = self._evaluate_model(int4_model, val_ds)

        binary_model = self._binarize_model()
        binary_accuracy = self._evaluate_model(binary_model, val_ds)

        results = {
            "int4_quantization": {
                "model": int4_model,
                "accuracy": int4_accuracy,
                "compression_ratio": self._estimate_compression_ratio(
                    default_bits=4
                ),
            },
            "binary_quantization": {
                "model": binary_model,
                "accuracy": binary_accuracy,
                "compression_ratio": self._estimate_compression_ratio(
                    default_bits=1
                ),
            },
            "accuracy_degradation_analysis": {
                "int4_drop": self._baseline_accuracy(val_ds) - int4_accuracy,
                "binary_drop": self._baseline_accuracy(val_ds) - binary_accuracy,
            },
            "performance_improvements": {
                "int4_speedup_estimate": 32 / 4,
                "binary_speedup_estimate": 32,
            },
        }

        self.quantization_results["extreme"] = results
        return results

    def dynamic_quantization_analysis(
        self,
        sample_batches: int = 10,
        batch_size: int = 256,
    ) -> Dict:
        """
        Compare static vs dynamic activation quantization on logits.
        """

        val_ds = self._get_dataset("val", batch_size=batch_size)
        static_range = [np.inf, -np.inf]
        dynamic_ranges: List[Tuple[float, float]] = []
        dynamic_correct, static_correct, total = 0, 0, 0

        for batch_idx, (images, labels) in enumerate(val_ds.take(sample_batches)):
            logits = self.base_model(images, training=False).numpy()
            batch_min = float(np.min(logits))
            batch_max = float(np.max(logits))
            dynamic_ranges.append((batch_min, batch_max))
            static_range[0] = min(static_range[0], batch_min)
            static_range[1] = max(static_range[1], batch_max)

            dynamic_quant = self._fake_quantize_array(logits, 8, batch_min, batch_max)
            static_quant = self._fake_quantize_array(
                logits, 8, static_range[0], static_range[1]
            )

            y_true = labels.numpy().flatten()
            dynamic_pred = np.argmax(dynamic_quant, axis=-1)
            static_pred = np.argmax(static_quant, axis=-1)
            dynamic_correct += int(np.sum(dynamic_pred == y_true))
            static_correct += int(np.sum(static_pred == y_true))
            total += y_true.size

        analysis = {
            "dynamic_accuracy": dynamic_correct / max(1, total),
            "static_accuracy": static_correct / max(1, total),
            "dynamic_ranges": dynamic_ranges,
            "static_range": tuple(static_range),
            "range_variability": float(
                np.std([r[1] - r[0] for r in dynamic_ranges])
                if dynamic_ranges
                else 0.0
            ),
        }
        self.quantization_results["dynamic_vs_static"] = analysis
        return analysis

    def quantization_error_analysis(
        self,
        bits: int = 8,
    ) -> Dict:
        """
        Measure weight-level quantization error per layer.
        """

        quant_layers = self._quantizable_layers(self.base_model)
        error_stats = {}
        for layer in quant_layers:
            weights = layer.get_weights()
            if not weights:
                continue
            quantized = [self._fake_quantize_array(w, bits) for w in weights]
            mse = float(
                np.mean(
                    [np.mean(np.square(q - w)) for q, w in zip(quantized, weights)]
                )
            )
            max_error = float(
                np.max([np.max(np.abs(q - w)) for q, w in zip(quantized, weights)])
            )
            error_stats[layer.name] = {
                "mse": mse,
                "max_abs_error": max_error,
                "sensitivity_proxy": self._estimate_layer_sensitivity(layer),
            }

        analysis = {
            "bits": bits,
            "layer_errors": error_stats,
            "most_sensitive_layers": sorted(
                error_stats.items(), key=lambda item: item[1]["mse"], reverse=True
            )[:5],
        }
        self.quantization_results["error_analysis"] = analysis
        return analysis

    # ------------------------------------------------------------------ #
    # Static helpers for other modules
    # ------------------------------------------------------------------ #
    @staticmethod
    def quantize_model(
        model: tf.keras.Model,
        bits: int = 8,
        bit_assignment: Optional[Dict[str, int]] = None,
    ) -> tf.keras.Model:
        """Expose quantization for other modules without instantiating the class."""

        pipeline = QuantizationPipeline(model, cache_datasets=False)
        return pipeline._quantize_model(default_bits=bits, bit_assignment=bit_assignment)

    # ------------------------------------------------------------------ #
    # Internal utilities
    # ------------------------------------------------------------------ #
    def _prepare_datasets(
        self,
        batch_size: int,
    ) -> DatasetBundle:
        (
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            calibration,
        ) = prepare_compression_datasets()

        def build_ds(x, y, augment=False):
            ds = tf.data.Dataset.from_tensor_slices((x, y))
            if augment:
                aug = tf.keras.Sequential(
                    [
                        tf.keras.layers.RandomFlip("horizontal"),
                        tf.keras.layers.RandomTranslation(0.1, 0.1),
                        tf.keras.layers.RandomRotation(0.05),
                    ]
                )

                def apply_aug(img, label):
                    return aug(img, training=True), label

                ds = ds.map(apply_aug, num_parallel_calls=AUTOTUNE)
            return ds.batch(batch_size).prefetch(AUTOTUNE)

        bundle = DatasetBundle(
            train=build_ds(x_train, y_train, augment=True),
            val=build_ds(x_val, y_val),
            test=build_ds(x_test, y_test),
            calibration=calibration,
            train_size=len(x_train),
            val_size=len(x_val),
            test_size=len(x_test),
        )
        return bundle

    def _get_dataset(
        self,
        split: str,
        batch_size: int,
    ) -> tf.data.Dataset:
        if (
            self._dataset_bundle is None
            or self._cached_batch_size != batch_size
            or not self.cache_datasets
        ):
            self._dataset_bundle = self._prepare_datasets(batch_size=batch_size)
            self._cached_batch_size = batch_size

        assert self._dataset_bundle is not None
        if split == "train":
            return self._dataset_bundle.train
        if split == "val":
            return self._dataset_bundle.val
        return self._dataset_bundle.test

    def _calibration_size(self) -> int:
        if self._dataset_bundle is None:
            self._dataset_bundle = self._prepare_datasets(batch_size=self._cached_batch_size)
        assert self._dataset_bundle is not None
        return len(self._dataset_bundle.calibration[0])

    def _quantizable_layers(
        self,
        model: tf.keras.Model,
    ) -> List[tf.keras.layers.Layer]:
        return [
            layer
            for layer in model.layers
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense))
            and layer.get_weights()
        ]

    def _estimate_layer_sensitivity(
        self,
        layer: tf.keras.layers.Layer,
    ) -> float:
        weights = layer.get_weights()
        if not weights:
            return 0.0
        kernel = weights[0]
        return float(np.mean(np.abs(kernel)))

    def _assign_bits(
        self,
        bit_configurations: List[int],
        sensitivity: Dict[str, float],
    ) -> Dict[str, int]:
        sorted_layers = sorted(
            sensitivity.items(), key=lambda item: item[1], reverse=True
        )
        num_segments = max(1, len(bit_configurations))
        assignment = {}
        total_layers = max(1, len(sorted_layers))
        for idx, (layer_name, _) in enumerate(sorted_layers):
            segment = min(idx * num_segments // total_layers, num_segments - 1)
            assignment[layer_name] = bit_configurations[segment]
        return assignment

    def _quantize_model(
        self,
        bit_assignment: Optional[Dict[str, int]] = None,
        default_bits: int = 8,
    ) -> tf.keras.Model:
        clone = tf.keras.models.clone_model(self.base_model)
        clone.set_weights(self.base_model.get_weights())

        optimizer = tf.keras.optimizers.deserialize(self._optimizer_config)
        loss = tf.keras.losses.deserialize(self._loss_config, custom_objects=CUSTOM_OBJECTS)
        # Use string metrics to avoid sample_weight parameter conflicts
        metrics = ["accuracy"]
        clone.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        for layer in self._quantizable_layers(clone):
            weights = layer.get_weights()
            if not weights:
                continue
            bits = bit_assignment.get(layer.name, default_bits) if bit_assignment else default_bits
            quantized = [self._fake_quantize_array(weight, bits) for weight in weights]
            layer.set_weights(quantized)

        return clone

    def _binarize_model(self) -> tf.keras.Model:
        clone = tf.keras.models.clone_model(self.base_model)
        clone.set_weights(self.base_model.get_weights())
        optimizer = tf.keras.optimizers.deserialize(self._optimizer_config)
        loss = tf.keras.losses.deserialize(self._loss_config, custom_objects=CUSTOM_OBJECTS)
        metrics = []
        for metric_config in self._metric_configs:
            if isinstance(metric_config, str):
                metrics.append(metric_config)
            else:
                try:
                    metrics.append(tf.keras.metrics.deserialize(metric_config, custom_objects=CUSTOM_OBJECTS))
                except Exception:
                    metrics.append(tf.keras.metrics.CategoricalAccuracy(name="accuracy"))
        clone.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        for layer in self._quantizable_layers(clone):
            weights = layer.get_weights()
            if not weights:
                continue
            binarized = []
            for weight in weights:
                scale = np.mean(np.abs(weight)) + 1e-7
                binarized.append(scale * np.sign(weight))
            layer.set_weights(binarized)
        return clone

    def _quantization_aware_finetune(
        self,
        model: tf.keras.Model,
        train_subset: tf.data.Dataset,
        epochs: int,
        steps_per_epoch: int,
        bits: int,
    ) -> Dict[str, List[float]]:
        history = {"loss": [], "accuracy": []}
        for epoch in range(epochs):
            hist = model.fit(
                train_subset,
                epochs=1,
                steps_per_epoch=steps_per_epoch,
                verbose=0,
            )
            history["loss"].append(hist.history["loss"][-1])
            history["accuracy"].append(hist.history.get("accuracy", [0.0])[-1])
            # Re-quantize weights after each epoch to emulate QAT.
            for layer in self._quantizable_layers(model):
                weights = layer.get_weights()
                if not weights:
                    continue
                quantized = [self._fake_quantize_array(weight, bits) for weight in weights]
                layer.set_weights(quantized)
        return history

    def _fake_quantize_array(
        self,
        array: np.ndarray,
        bits: int,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> np.ndarray:
        array = array.astype(np.float32)
        levels = max(2, 2**bits - 1)
        if min_value is None or max_value is None:
            min_value = float(np.min(array))
            max_value = float(np.max(array))
        if math.isclose(max_value, min_value):
            return array
        scale = (max_value - min_value) / levels
        quantized = np.round((array - min_value) / scale)
        dequantized = quantized * scale + min_value
        return dequantized.astype(np.float32)

    def _evaluate_model(
        self,
        model: tf.keras.Model,
        dataset: tf.data.Dataset,
    ) -> float:
        """Evaluate model with workaround for tf_keras sample_weight issue."""
        try:
            # Try standard evaluation first
            metrics = model.evaluate(dataset, verbose=0)
            if isinstance(metrics, list):
                return float(metrics[1]) if len(metrics) > 1 else float(metrics[0])
            return float(metrics)
        except TypeError as e:
            if "sample_weight" in str(e):
                # Fallback: manual evaluation for sample_weight conflicts
                total_correct = 0
                total_samples = 0
                for x_batch, y_batch in dataset:
                    predictions = model.predict(x_batch, verbose=0)
                    predicted_labels = tf.argmax(predictions, axis=1)
                    true_labels = tf.argmax(y_batch, axis=1) if len(y_batch.shape) > 1 else y_batch
                    total_correct += tf.reduce_sum(
                        tf.cast(tf.equal(predicted_labels, true_labels), tf.int32)
                    ).numpy()
                    total_samples += x_batch.shape[0]
                return float(total_correct / total_samples) if total_samples > 0 else 0.0
            raise

    def _baseline_accuracy(self, dataset: tf.data.Dataset) -> float:
        if "baseline_accuracy" not in self.quantization_results:
            acc = self._evaluate_model(self.base_model, dataset)
            self.quantization_results["baseline_accuracy"] = {"accuracy": acc}
        return self.quantization_results["baseline_accuracy"]["accuracy"]

    def _estimate_compression_ratio(
        self,
        bit_assignment: Optional[Dict[str, int]] = None,
        default_bits: int = 8,
    ) -> float:
        total_params = 0
        encoded_bits = 0
        for layer in self._quantizable_layers(self.base_model):
            weights = layer.get_weights()
            if not weights:
                continue
            params = sum(weight.size for weight in weights)
            bits = bit_assignment.get(layer.name, default_bits) if bit_assignment else default_bits
            total_params += params
            encoded_bits += params * bits
        if total_params == 0:
            return 1.0
        baseline_bits = total_params * 32
        return baseline_bits / max(1, encoded_bits)
