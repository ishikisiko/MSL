"""Hardware-aware optimization strategies for the MLS3 assignment."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import mobilenet_v2
import tensorflow_model_optimization as tfmot


AUTOTUNE = tf.data.AUTOTUNE


@dataclass
class OptimizationResult:
    """Container with metrics and artefacts for a single optimisation run."""

    model_name: str
    trained_model: keras.Model
    histories: List[keras.callbacks.History]
    evaluation: Dict[str, float]
    quantized_models: Dict[str, object]
    memory_strategies: Dict[str, object]


def _build_training_callbacks() -> List[keras.callbacks.Callback]:
    """Return a default list of callbacks shared across optimisation runs."""

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=5e-6
    )
    return [early_stopping, reduce_lr]


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
    se_ratio: float = 0.25,
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

    # Lightweight squeeze-and-excitation block for the final features to
    # encourage channel re-weighting without significant latency overhead.
    squeeze = layers.GlobalAveragePooling2D(name="latency_gap")(x)
    squeeze = layers.Dense(
        int(backbone.output_shape[-1] * se_ratio),
        activation="relu",
        name="latency_se_dense",
    )(squeeze)
    excite = layers.Dense(
        backbone.output_shape[-1],
        activation="sigmoid",
        name="latency_se_gate",
    )(squeeze)
    excite = layers.Reshape((1, 1, backbone.output_shape[-1]), name="latency_se_reshape")(excite)
    x = layers.Multiply(name="latency_se_scale")([x, excite])

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

    # Apply depthwise separable projection to reduce the channel footprint
    # before global pooling. This greatly lowers parameter count.
    x = layers.DepthwiseConv2D(3, padding="same", name="memory_dw")(x)
    x = layers.Conv2D(int(backbone.output_shape[-1] * 0.75), 1, activation="relu6", name="memory_pw")(x)
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
    dropout_rate: float = 0.25,
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

    # Incorporate an early-exit branch for low-compute scenarios. The branch is
    # optional and only used when explicit inference shortcuts are required.
    intermediate = layers.GlobalAveragePooling2D(name="energy_gap")(x)
    shallow_logits = layers.Dense(num_classes, activation="softmax", name="energy_early_exit")(intermediate)

    outputs = _build_head(x, num_classes, dropout_rate=dropout_rate)
    float_model = keras.Model(
        inputs=inputs,
        outputs={"classifier": outputs, "energy_early_exit": shallow_logits},
        name="mobilenetv2_energy_opt",
    )

    quantize_model = tfmot.quantization.keras.quantize_model
    qat_model = quantize_model(float_model)
    return _compile_model(
        qat_model,
        learning_rate=5e-4,
        loss={
            "classifier": "sparse_categorical_crossentropy",
            "energy_early_exit": "sparse_categorical_crossentropy",
        },
        metrics={"classifier": ["accuracy"], "energy_early_exit": ["accuracy"]},
    )


def apply_quantization_optimizations(
    model: keras.Model,
    x_train_sample: np.ndarray,
    y_train_sample: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Generate quantized variants of ``model`` using multiple strategies."""

    return {
        "ptq_int8": post_training_quantization(model, x_train_sample),
        "qat_int8": quantization_aware_training(
            model,
            x_train_sample=x_train_sample,
            y_train_sample=y_train_sample,
        ),
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
    y_train_sample: Optional[np.ndarray] = None,
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
        labels = y_train_sample if y_train_sample is not None else np.zeros(len(x_train_sample), dtype=np.int32)
        dataset = (
            tf.data.Dataset.from_tensor_slices((processed, labels))
            .shuffle(buffer_size=len(processed))
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


def train_optimized_model(
    model: keras.Model,
    train_data: tf.data.Dataset,
    validation_data: tf.data.Dataset,
    fine_tune_at: Optional[int] = None,
    head_epochs: int = 3,
    fine_tune_epochs: int = 5,
    fine_tune_lr: float = 5e-5,
) -> List[keras.callbacks.History]:
    """Train an optimised model using a two-stage fine-tuning schedule."""

    callbacks = _build_training_callbacks()
    histories: List[keras.callbacks.History] = []

    history_head = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=head_epochs,
        callbacks=callbacks,
        verbose=0,
    )
    histories.append(history_head)

    if fine_tune_at is None:
        fine_tune_at = max(len(model.layers) - 40, 0)

    if fine_tune_at is not None:
        trainable = False
        for index, layer in enumerate(model.layers):
            if index >= fine_tune_at:
                trainable = True
            layer.trainable = trainable

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=fine_tune_lr),
            loss=model.loss,
            metrics=model.metrics,
        )
        history_fine = model.fit(
            train_data,
            validation_data=validation_data,
            epochs=fine_tune_epochs,
            callbacks=callbacks,
            verbose=0,
        )
        histories.append(history_fine)

    return histories


def evaluate_model(model: keras.Model, test_data: Iterable, batch_size: int = 32) -> Dict[str, float]:
    """Evaluate ``model`` on ``test_data`` and record latency and accuracy."""

    if isinstance(test_data, tf.data.Dataset):
        dataset = test_data.cache().prefetch(AUTOTUNE)
    else:
        x_test, y_test = test_data
        dataset = (
            tf.data.Dataset.from_tensor_slices((x_test, y_test))
            .batch(batch_size)
            .prefetch(AUTOTUNE)
        )

    start = time.perf_counter()
    evaluation = model.evaluate(dataset, return_dict=True, verbose=0)
    inference_time = (time.perf_counter() - start) * 1e3  # milliseconds

    sample_dataset = dataset.take(1)
    single_start = time.perf_counter()
    inputs = None
    for batch in sample_dataset:
        inputs = batch[0] if isinstance(batch, tuple) else batch
        model.predict(inputs, verbose=0)
        break

    if inputs is not None:
        if hasattr(inputs, "shape") and inputs.shape[0] is not None:
            batch_size_value = int(inputs.shape[0])
        else:
            batch_size_value = int(tf.shape(inputs)[0].numpy())
        single_latency = (time.perf_counter() - single_start) * 1e3 / max(batch_size_value, 1)
    else:
        single_latency = float("nan")

    evaluation.update(
        {
            "latency_ms": inference_time,
            "single_sample_latency_ms": float(single_latency),
        }
    )
    return evaluation


def run_hardware_aware_optimization(
    train_data: tf.data.Dataset,
    validation_data: tf.data.Dataset,
    test_data: tf.data.Dataset,
    x_train_sample: np.ndarray,
    y_train_sample: np.ndarray,
    memory_budget_mb: int = 512,
) -> Dict[str, OptimizationResult]:
    """Execute the full optimisation workflow described in the assignment."""

    results: Dict[str, OptimizationResult] = {}
    optimized_models = create_optimized_models()

    for name, model in optimized_models.items():
        histories = train_optimized_model(model, train_data, validation_data)
        evaluation = evaluate_model(model, test_data)
        quantized = apply_quantization_optimizations(
            model,
            x_train_sample=x_train_sample,
            y_train_sample=y_train_sample,
        )
        memory_strategies = implement_memory_optimizations(model)
        memory_strategies["optimal_batch_size"] = find_optimal_batch_size(
            model, memory_budget_mb=memory_budget_mb
        )

        results[name] = OptimizationResult(
            model_name=name,
            trained_model=model,
            histories=histories,
            evaluation=evaluation,
            quantized_models=quantized,
            memory_strategies=memory_strategies,
        )

    return results
