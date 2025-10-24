"""Baseline MobileNetV2 implementation for the MLS3 assignment."""

from __future__ import annotations

import os
import time
from typing import Dict, Iterable, Tuple

import numpy as np
import psutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2_as_graph,
)


AUTOTUNE = tf.data.AUTOTUNE


def _build_data_augmentation() -> keras.Sequential:
    """Create the data-augmentation pipeline used for training."""

    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )


def create_baseline_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 10,
    base_learning_rate: float = 1e-3,
) -> keras.Model:
    """Create and compile a MobileNetV2-based classifier.

    The function constructs a MobileNetV2 backbone pre-trained on ImageNet and
    augments it with a small task-specific classification head. The backbone is
    frozen by default to facilitate transfer learning, allowing callers to
    perform head-only training before unfreezing for fine-tuning.

    Args:
        input_shape: Input dimensions for the classifier.
        num_classes: Number of target classes.
        base_learning_rate: Optimizer learning rate used for the initial phase.

    Returns:
        A compiled ``tf.keras.Model`` ready for training.
    """

    inputs = keras.Input(shape=input_shape, name="input_image")
    augmentation = _build_data_augmentation()
    x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale_inputs")(x)

    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling=None,
    )
    backbone.trainable = False

    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="baseline_mobilenetv2")

    optimizer = keras.optimizers.Adam(learning_rate=base_learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_and_preprocess_data(
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 64,
    validation_split: float = 0.1,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load CIFAR-10 and preprocess samples for MobileNetV2.

    The routine performs the following steps:

    * loads CIFAR-10 from ``keras.datasets``;
    * scales images to floating point and resizes them to ``image_size``;
    * applies light data augmentation for the training split;
    * builds ``tf.data.Dataset`` pipelines for train/validation/test sets.

    Args:
        image_size: Target spatial resolution (height, width).
        batch_size: Number of samples per batch.
        validation_split: Fraction of training samples reserved for validation.

    Returns:
        A tuple of ``(train_ds, val_ds, test_ds)`` datasets ready for training
        and evaluation.
    """

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    val_size = int(len(x_train) * validation_split)
    if val_size == 0:
        raise ValueError("validation_split is too small; no validation samples available.")

    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]

    preprocess_fn = mobilenet_v2.preprocess_input
    augmenter = _build_data_augmentation()

    def _prepare_dataset(images: np.ndarray, labels: np.ndarray, augment: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        if augment:
            ds = ds.shuffle(buffer_size=len(images), reshuffle_each_iteration=True)

        def _preprocess(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            image = tf.image.resize(image, image_size)
            image = preprocess_fn(image)
            if augment:
                image = augmenter(image, training=True)
            return image, tf.squeeze(label, axis=-1)

        ds = ds.map(_preprocess, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
        return ds

    train_ds = _prepare_dataset(x_train, y_train, augment=True)
    val_ds = _prepare_dataset(x_val, y_val, augment=False)
    test_ds = _prepare_dataset(x_test, y_test, augment=False)

    return train_ds, val_ds, test_ds


def _calculate_model_size_mb(model: keras.Model) -> float:
    """Estimate the serialized model size in megabytes."""

    size_bytes = 0
    for weight in model.weights:
        size_bytes += np.prod(weight.shape) * weight.dtype.size
    return size_bytes / (1024 ** 2)


def _calculate_model_flops(model: keras.Model, batch_size: int = 1) -> int:
    """Calculate the number of floating-point operations for a forward pass."""

    try:
        inputs = tf.TensorSpec([batch_size, *model.input_shape[1:]], tf.float32)
        concrete_fn = tf.function(model).get_concrete_function(inputs)
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_fn)
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name="")
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(
                graph=graph,
                run_meta=run_meta,
                cmd="op",
                options=opts,
            )
        return flops.total_float_ops if flops is not None else 0
    except (ValueError, TypeError, AttributeError):
        # As a conservative fallback, return zero and allow callers to handle it.
        return 0


def benchmark_baseline_model(
    model: keras.Model,
    test_data: Iterable,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Benchmark the baseline model for latency, memory usage, and accuracy."""

    if isinstance(test_data, tf.data.Dataset):
        dataset = test_data
    else:
        x_test, y_test = test_data
        dataset = (
            tf.data.Dataset.from_tensor_slices((x_test, y_test))
            .batch(batch_size)
            .prefetch(AUTOTUNE)
        )

    dataset = dataset.cache().prefetch(AUTOTUNE)

    process = psutil.Process(os.getpid())

    # Extract a single sample for detailed latency probing.
    sample_batch = next(iter(dataset))
    sample_inputs = sample_batch[0][:1]

    # Warm-up to stabilise lazy initialisation overheads.
    _ = model.predict(sample_inputs, verbose=0)

    start = time.perf_counter()
    _ = model.predict(sample_inputs, verbose=0)
    single_inference_time = time.perf_counter() - start

    before_mem = process.memory_info().rss
    start = time.perf_counter()
    evaluation = model.evaluate(dataset, verbose=0, return_dict=True)
    batch_inference_time = time.perf_counter() - start
    after_mem = process.memory_info().rss

    predictions = model.predict(dataset, verbose=0)
    labels = np.concatenate([batch[1].numpy() for batch in dataset], axis=0)
    accuracy = float(np.mean(np.argmax(predictions, axis=-1) == labels))

    metrics: Dict[str, float] = {
        "single_inference_time": float(single_inference_time),
        "batch_inference_time": float(batch_inference_time),
        "memory_usage_mb": float(max(after_mem - before_mem, 0) / (1024 ** 2)),
        "model_size_mb": float(_calculate_model_size_mb(model)),
        "accuracy": accuracy,
        "flops": float(_calculate_model_flops(model, batch_size=1)),
        "parameters": float(model.count_params()),
    }

    metrics.update({f"eval_{k}": float(v) for k, v in evaluation.items()})
    return metrics


# --- 主要训练流程 ---

if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_and_preprocess_data()
    model = create_baseline_model()

    # Example training routine (commented to avoid accidental execution).
    # history = model.fit(train_ds, validation_data=val_ds, epochs=5)

    model.save("baseline_mobilenetv2.keras")

    metrics = benchmark_baseline_model(model, test_ds)
    print("Baseline Model Performance:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
