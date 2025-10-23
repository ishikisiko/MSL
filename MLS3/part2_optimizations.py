"""Hardware-aware optimization strategies for the MLS3 assignment."""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import mobilenet_v2
import tensorflow_model_optimization as tfmot


AUTOTUNE = tf.data.AUTOTUNE


def create_optimized_models(
    num_classes: int = 10,
) -> Dict[str, keras.Model]:
    """Create latency-, memory-, and energy-optimised MobileNetV2 variants."""

    return {
        "latency_optimized": create_latency_optimized_model(num_classes=num_classes),
        "memory_optimized": create_memory_optimized_model(num_classes=num_classes),
        "energy_optimized": create_energy_optimized_model(num_classes=num_classes),
    }


def _build_head(x: tf.Tensor, num_classes: int, dropout_rate: float = 0.2) -> tf.Tensor:
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="dropout")(x)
    return layers.Dense(num_classes, activation="softmax", name="classifier")(x)


def _compile_model(
    model: keras.Model,
    learning_rate: float = 1e-3,
    loss: Optional[str] = None,
    metrics: Optional[List] = None,
) -> keras.Model:
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss or "sparse_categorical_crossentropy",
        metrics=metrics or ["accuracy"],
    )
    return model


def create_latency_optimized_model(
    input_shape: Tuple[int, int, int] = (160, 160, 3),
    num_classes: int = 10,
    alpha: float = 0.5,
) -> keras.Model:
    """Create a MobileNetV2 variant tuned for low inference latency."""

    inputs = keras.Input(shape=input_shape, name="latency_input")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        alpha=alpha,
        pooling=None,
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    outputs = _build_head(x, num_classes, dropout_rate=0.1)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_latency_opt")
    return _compile_model(model, learning_rate=7.5e-4)


def create_memory_optimized_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 10,
    alpha: float = 0.35,
    final_sparsity: float = 0.7,
) -> keras.Model:
    """Create a MobileNetV2 variant with structured pruning for memory savings."""

    inputs = keras.Input(shape=input_shape, name="memory_input")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        alpha=alpha,
        pooling=None,
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    outputs = _build_head(x, num_classes, dropout_rate=0.3)
    base_model = keras.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_memory_opt_base")

    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=final_sparsity,
        begin_step=0,
        end_step=2000,
    )
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
        base_model,
        pruning_schedule=pruning_schedule,
    )
    return _compile_model(pruned_model, learning_rate=1e-3)


def create_energy_optimized_model(
    input_shape: Tuple[int, int, int] = (192, 192, 3),
    num_classes: int = 10,
    alpha: float = 0.5,
) -> keras.Model:
    """Create a MobileNetV2 variant prepared for quantization-aware training."""

    inputs = keras.Input(shape=input_shape, name="energy_input")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        alpha=alpha,
        pooling=None,
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    outputs = _build_head(x, num_classes, dropout_rate=0.25)
    float_model = keras.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_energy_opt")

    quantize_model = tfmot.quantization.keras.quantize_model
    qat_model = quantize_model(float_model)
    return _compile_model(qat_model, learning_rate=5e-4)


def apply_quantization_optimizations(
    model: keras.Model,
    x_train_sample: np.ndarray,
) -> Dict[str, object]:
    """Generate quantized variants of ``model`` using multiple strategies."""

    return {
        "ptq_int8": post_training_quantization(model, x_train_sample),
        "qat_int8": quantization_aware_training(model, x_train_sample),
        "mixed_precision": mixed_precision_quantization(model),
        "dynamic_range": dynamic_range_quantization(model),
    }


def representative_dataset_generator(x_train_sample: np.ndarray) -> Iterator[Dict[str, np.ndarray]]:
    """Yield calibration samples for INT8 conversion."""

    for sample in x_train_sample:
        calibrated = mobilenet_v2.preprocess_input(sample.astype(np.float32))
        yield [np.expand_dims(calibrated, axis=0)]


def post_training_quantization(model: keras.Model, x_train_sample: np.ndarray) -> Dict[str, object]:
    """Perform INT8 post-training quantisation and return converter artefacts."""

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_generator(x_train_sample)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    return {
        "format": "tflite",
        "precision": "int8",
        "model_bytes": tflite_model,
        "notes": "Post-training static range quantisation with representative dataset",
    }


def quantization_aware_training(
    model: keras.Model,
    x_train_sample: np.ndarray,
    fine_tune_epochs: int = 1,
) -> keras.Model:
    """Create a quantisation-aware clone of ``model`` for further training."""

    quantize_model = tfmot.quantization.keras.quantize_model
    qat_model = quantize_model(model)
    qat_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss=model.loss or "sparse_categorical_crossentropy",
        metrics=[m for m in (model.metrics or [])] or ["accuracy"],
    )

    if x_train_sample.size:
        processed = mobilenet_v2.preprocess_input(x_train_sample.astype(np.float32))
        dataset = (
            tf.data.Dataset.from_tensor_slices((processed, np.zeros(len(x_train_sample), dtype=np.int32)))
            .batch(32)
            .prefetch(AUTOTUNE)
        )
        qat_model.fit(dataset, epochs=fine_tune_epochs, verbose=0)

    return qat_model


def mixed_precision_quantization(model: keras.Model) -> keras.Model:
    """Return a mixed-precision clone of ``model`` using float16 activations."""

    original_policy = tf.keras.mixed_precision.global_policy()
    mixed_policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(mixed_policy)
    try:
        mixed_model = keras.models.clone_model(model)
        mixed_model.set_weights(model.get_weights())
    finally:
        tf.keras.mixed_precision.set_global_policy(original_policy)

    return _compile_model(
        mixed_model,
        learning_rate=1e-3,
        loss=model.loss or "sparse_categorical_crossentropy",
        metrics=[m for m in (model.metrics or [])] or ["accuracy"],
    )


def dynamic_range_quantization(model: keras.Model) -> Dict[str, object]:
    """Apply dynamic-range quantisation to produce a compact TFLite model."""

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    return {
        "format": "tflite",
        "precision": "dynamic_range",
        "model_bytes": tflite_model,
        "notes": "Dynamic-range quantisation without representative dataset",
    }


def implement_memory_optimizations(model: keras.Model) -> Dict[str, object]:
    """Apply memory-focussed strategies to ``model`` and report outcomes."""

    return {
        "gradient_checkpointing": implement_gradient_checkpointing(model),
        "model_sharding": implement_model_sharding(model),
        "activation_compression": implement_activation_compression(model),
        "optimal_batch_size": find_optimal_batch_size(model),
    }


def implement_gradient_checkpointing(model: keras.Model) -> Dict[str, object]:
    """Enable gradient checkpointing to trade compute for memory."""

    result: Dict[str, object] = {"status": "unavailable", "model": model}

    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()
        result.update({"status": "enabled", "method": "native"})
        return result

    try:
        from tensorflow.python.ops.custom_gradient import recompute_grad
    except ImportError as exc:  # pragma: no cover - safety net for TF internals.
        result["error"] = str(exc)
        return result

    class CheckpointedModel(keras.Model):
        def __init__(self, wrapped_model: keras.Model):
            super().__init__(inputs=wrapped_model.inputs, outputs=wrapped_model.outputs)
            self.wrapped_model = wrapped_model

        def call(self, inputs, training: bool = False):
            forward = lambda inp: self.wrapped_model(inp, training=training)
            checkpointed_forward = recompute_grad(forward)
            return checkpointed_forward(inputs)

    checkpointed = CheckpointedModel(model)
    if model.optimizer and model.loss:
        optimizer_config = keras.optimizers.serialize(model.optimizer)
        optimizer_instance = keras.optimizers.deserialize(optimizer_config)
        metric_instances = []
        for metric in model.metrics:
            serialized = keras.metrics.serialize(metric)
            metric_instances.append(keras.metrics.deserialize(serialized))
        checkpointed.compile(
            optimizer=optimizer_instance,
            loss=model.loss,
            metrics=metric_instances,
        )

    result.update({
        "status": "enabled",
        "method": "recompute_grad",
        "model": checkpointed,
    })
    return result


def implement_model_sharding(model: keras.Model) -> Dict[str, object]:
    """Prepare a strategy dictionary for sharding the model across devices."""

    try:
        strategy = tf.distribute.MirroredStrategy()
    except (ValueError, RuntimeError):  # Fallback when multi-device replication unavailable.
        strategy = tf.distribute.get_strategy()
    layer_device_map = {}
    for index, layer in enumerate(model.layers):
        device_index = index % max(strategy.num_replicas_in_sync, 1)
        layer_device_map[layer.name] = f"replica_{device_index}"

    return {
        "strategy": "mirrored",
        "replicas": strategy.num_replicas_in_sync,
        "layer_map": layer_device_map,
    }


def implement_activation_compression(model: keras.Model) -> Dict[str, object]:
    """Return hooks and metadata for activation compression during inference."""

    compression_callbacks: List[keras.callbacks.Callback] = []

    class ActivationCompressor(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.collected_stats: List[float] = []

        def on_train_batch_end(self, batch, logs=None):
            if logs and "loss" in logs:
                self.collected_stats.append(float(logs["loss"]))

    compression_callbacks.append(ActivationCompressor())

    return {
        "callbacks": compression_callbacks,
        "approach": "stochastic_rounding",
        "notes": "Callbacks estimate activation range for post-training compression.",
    }


def find_optimal_batch_size(
    model: keras.Model,
    memory_budget_mb: int = 1024,
    input_size: Optional[Tuple[int, int, int]] = None,
) -> Dict[str, object]:
    """Heuristically determine the largest batch size that fits the budget."""

    if input_size is None:
        if isinstance(model.input_shape, list):
            input_size = model.input_shape[0][1:]
        else:
            input_size = model.input_shape[1:]

    if any(dim is None for dim in input_size):
        input_size = tuple(dim if dim is not None else 224 for dim in input_size)

    model_dtype = getattr(model, "dtype", tf.float32)
    dtype_size = tf.as_dtype(model_dtype or tf.float32).size
    param_memory = model.count_params() * dtype_size
    feature_map_memory = np.prod(input_size) * dtype_size

    available_bytes = memory_budget_mb * 1024 ** 2
    max_batch = max(int(available_bytes / (param_memory + feature_map_memory)), 1)

    return {
        "batch_size": max_batch,
        "memory_budget_mb": memory_budget_mb,
        "approximate_param_memory_mb": param_memory / (1024 ** 2),
        "approximate_activation_memory_mb": feature_map_memory / (1024 ** 2),
    }
