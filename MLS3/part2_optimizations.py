"""Hardware-aware optimizations for the MLS3 assignment.

This module provides helper functions to create latency-, memory- and
energy-optimized variants from the baseline MobileNetV2 and utilities to
apply quantization (PTQ/QAT/mixed) and simple memory optimizations.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

try:
    import tensorflow_model_optimization as tfmot
    TF_MOT_AVAILABLE = True
except Exception:
    TF_MOT_AVAILABLE = False

from tensorflow.keras import layers


def create_latency_optimized_model(input_shape=(128, 128, 3), num_classes=10, alpha: float = 0.5):
    """Create a MobileNetV2 variant optimized for latency.

    - Uses a reduced depth multiplier (alpha)
    - Lower input resolution by default (128x128)
    - Smaller Dropout and lighter head
    """
    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet", alpha=alpha
    )
    x = backbone.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.15)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=backbone.input, outputs=outputs, name=f"mobilenetv2_latency_alpha{alpha}")
    return model


def create_memory_optimized_model(input_shape=(96, 96, 3), num_classes=10, width_multiplier: float = 0.5):
    """Create a memory-optimized model by reducing channels and (optionally) applying pruning.

    This function returns a smaller MobileNetV2 backbone and a flag indicating
    whether pruning can be applied (depends on tfmot availability).
    """
    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet", alpha=width_multiplier
    )
    x = backbone.output
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=backbone.input, outputs=outputs, name=f"mobilenetv2_memory_w{width_multiplier}")

    if TF_MOT_AVAILABLE:
        # Return a function that can apply pruning; do not prune by default here to keep model usable.
        return model
    else:
        return model


def create_energy_optimized_model(input_shape=(96, 96, 3), num_classes=10):
    """Create an energy-optimized model skeleton.

    Typical strategy: enable lower-precision compute (handled later in conversion) and
    design a shallow head / early-exit strategies (not fully implemented here).
    """
    backbone = keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet", alpha=0.75)
    x = backbone.output
    x = layers.GlobalAveragePooling2D()(x)
    # Small head to reduce compute
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=backbone.input, outputs=outputs, name="mobilenetv2_energy_opt")
    return model


def representative_dataset_generator(x_samples: Iterable[np.ndarray], num_samples: int = 200):
    """Yield representative samples for TFLite quantization calibration.

    Arguments:
        x_samples: Iterable/array of input images (uint8/float32)
        num_samples: max number of samples to yield
    """
    count = 0
    for sample in x_samples:
        if count >= num_samples:
            break
        # If sample is a batch, yield each element
        if hasattr(sample, "shape") and sample.shape[0] > 1:
            for s in sample:
                yield np.expand_dims(s.astype(np.float32), axis=0)
                count += 1
                if count >= num_samples:
                    break
        else:
            yield np.expand_dims(np.array(sample, dtype=np.float32), axis=0)
            count += 1


def post_training_quantization(model: keras.Model, representative_data: Iterable[np.ndarray], save_path: str = "model_ptq.tflite") -> str:
    """Perform a simple post-training full integer quantization to TFLite.

    Returns path to the saved TFLite model.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Ensure input/output are int8
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(save_path, "wb") as f:
        f.write(tflite_model)
    return save_path


def quantization_aware_training(model: keras.Model, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset, epochs: int = 5) -> keras.Model:
    """Apply a QAT workflow using TF-MOT if available.

    If TF-MOT is not available, return the original model (no-op).
    """
    if not TF_MOT_AVAILABLE:
        print("TF Model Optimization Toolkit not available; skipping QAT.")
        return model

    quantize_scope = tfmot.quantization.keras.quantize_scope
    QuantizeConfig = tfmot.quantization.keras.quantize_config

    quantize_model = tfmot.quantization.keras.quantize_model

    q_model = quantize_model(model)
    q_model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"]) 
    q_model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, verbose=1)
    return q_model


def mixed_precision_quantization(model: keras.Model) -> keras.Model:
    """Prepare model for mixed precision training/inference.

    This function only sets the policy; actual benefits depend on hardware and runtime.
    """
    try:
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy("mixed_float16")
        return model
    except Exception:
        print("Mixed-precision not available in this TF build; returning original model.")
        return model


def dynamic_range_quantization(model: keras.Model, save_path: str = "model_dynamic.tflite") -> str:
    """Apply dynamic-range quantization (weights quantized) and save TFLite file."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(save_path, "wb") as f:
        f.write(tflite_model)
    return save_path


def implement_gradient_checkpointing(model: keras.Model) -> keras.Model:
    """Placeholder for gradient checkpointing.

    Full gradient checkpointing requires model re-wiring or third-party libraries.
    Return the model unchanged and a note.
    """
    print("Gradient checkpointing: placeholder — use tf.recompute_grad or custom training loop for real effect.")
    return model


def find_optimal_batch_size(model: keras.Model, start: int = 1, max_batch: int = 128) -> int:
    """Heuristic search for largest batch that fits in memory (best-effort).

    This performs a naive trial by increasing batch size until OOM is observed.
    """
    import gc

    for b in [1, 2, 4, 8, 16, 32, 64, 128]:
        if b < start or b > max_batch:
            continue
        try:
            dummy = np.zeros((b, *model.input_shape[1:]), dtype=np.float32)
            _ = model.predict(dummy, verbose=0)
            gc.collect()
        except Exception:
            return max(b // 2, 1)
    return max_batch


__all__ = [
    "create_latency_optimized_model",
    "create_memory_optimized_model",
    "create_energy_optimized_model",
    "representative_dataset_generator",
    "post_training_quantization",
    "quantization_aware_training",
    "mixed_precision_quantization",
    "dynamic_range_quantization",
    "implement_gradient_checkpointing",
    "find_optimal_batch_size",
]
