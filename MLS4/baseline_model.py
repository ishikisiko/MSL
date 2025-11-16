from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE
CALIBRATION_SIZE = 1000
VAL_SIZE = 5000
DEFAULT_MODEL_PATH = "baseline_model.keras"
DEFAULT_CHECKPOINT_PATH = "checkpoints/baseline_best.keras"


@dataclass
class DatasetSplits:
    """Container for CIFAR-100 datasets prepared for compression experiments."""

    train: tf.data.Dataset
    val: tf.data.Dataset
    test: tf.data.Dataset
    calibration: tf.data.Dataset
    train_size: int
    val_size: int
    test_size: int


def _to_int_labels(labels: tf.Tensor) -> tf.Tensor:
    if not labels.dtype.is_integer:
        labels = tf.cast(tf.round(labels), tf.int32)
    else:
        labels = tf.cast(labels, tf.int32)
    return labels


def _ensure_one_hot_labels(y_true: tf.Tensor, num_classes: int, dtype: tf.dtypes.DType) -> tf.Tensor:
    """Convert sparse or 1-hot labels to a consistent dense 1-hot representation."""

    y_true = tf.convert_to_tensor(y_true)
    dtype = tf.as_dtype(dtype)
    static_rank = y_true.shape.rank
    static_last_dim = y_true.shape[-1] if static_rank and static_rank > 0 else None

    if static_rank is not None and static_rank > 0 and static_last_dim == num_classes:
        return tf.cast(y_true, dtype)

    if static_rank is not None and static_rank > 0 and static_last_dim == 1:
        squeezed = tf.squeeze(y_true, axis=-1)
        return tf.one_hot(_to_int_labels(squeezed), depth=num_classes, dtype=dtype)

    if static_rank == 0:
        y_true = tf.expand_dims(y_true, axis=0)
        return tf.one_hot(_to_int_labels(y_true), depth=num_classes, dtype=dtype)

    if static_rank is not None:
        return tf.one_hot(_to_int_labels(y_true), depth=num_classes, dtype=dtype)

    # Fallback for tensors with dynamic shape information.
    shape = tf.shape(y_true)
    last_dim = shape[-1]

    def cast_as_one_hot() -> tf.Tensor:
        return tf.cast(y_true, dtype)

    def convert_sparse() -> tf.Tensor:
        def squeeze_if_needed() -> tf.Tensor:
            return tf.squeeze(y_true, axis=-1)

        def identity() -> tf.Tensor:
            return y_true

        adjusted = tf.cond(tf.equal(last_dim, 1), squeeze_if_needed, identity)
        return tf.one_hot(_to_int_labels(adjusted), depth=num_classes, dtype=dtype)

    return tf.cond(tf.equal(last_dim, num_classes), cast_as_one_hot, convert_sparse)


@tf.keras.utils.register_keras_serializable(package="mls4")
class AdaptiveCategoricalCrossentropy(tf.keras.losses.Loss):
    """Cross-entropy that accepts either sparse or one-hot labels with smoothing."""

    def __init__(self, num_classes: int, label_smoothing: float = 0.0, **kwargs) -> None:
        name = kwargs.pop("name", "adaptive_categorical_crossentropy")
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.label_smoothing = float(label_smoothing)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = _ensure_one_hot_labels(y_true, self.num_classes, y_pred.dtype)
        if self.label_smoothing:
            smoothing = tf.cast(self.label_smoothing, y_pred.dtype)
            classes = tf.cast(self.num_classes, y_pred.dtype)
            y_true = y_true * (1.0 - smoothing) + smoothing / classes

        y_pred = tf.clip_by_value(
            y_pred,
            clip_value_min=tf.keras.backend.epsilon(),
            clip_value_max=1.0 - tf.keras.backend.epsilon(),
        )
        per_example_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        return per_example_loss

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "label_smoothing": self.label_smoothing,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="mls4")
class AdaptiveCategoricalAccuracy(tf.keras.metrics.CategoricalAccuracy):
    """Categorical accuracy metric resilient to sparse labels."""

    def __init__(self, num_classes: int, name: str = "accuracy", **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: Optional[tf.Tensor] = None) -> None:
        y_true = _ensure_one_hot_labels(y_true, self.num_classes, y_pred.dtype)
        return super().update_state(y_true, y_pred, sample_weight)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config


@tf.keras.utils.register_keras_serializable(package="mls4")
class AdaptiveTopKCategoricalAccuracy(tf.keras.metrics.TopKCategoricalAccuracy):
    """Top-K accuracy metric that accepts either sparse or one-hot labels."""

    def __init__(self, num_classes: int, k: int = 5, name: str = "top5", **kwargs) -> None:
        super().__init__(k=k, name=name, **kwargs)
        self.num_classes = num_classes

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: Optional[tf.Tensor] = None) -> None:
        y_true = _ensure_one_hot_labels(y_true, self.num_classes, y_pred.dtype)
        return super().update_state(y_true, y_pred, sample_weight)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config


CUSTOM_OBJECTS: Dict[str, Any] = {
    "AdaptiveCategoricalCrossentropy": AdaptiveCategoricalCrossentropy,
    "AdaptiveCategoricalAccuracy": AdaptiveCategoricalAccuracy,
    "AdaptiveTopKCategoricalAccuracy": AdaptiveTopKCategoricalAccuracy,
}


def create_baseline_model(
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 100,
    dropout_rate: float = 0.2,
    width_multiplier: float = 1.0,
) -> tf.keras.Model:
    """
    Create a lightweight ResNet-style baseline suitable for CIFAR-100.

    The architecture intentionally mirrors Keras' ResNet blocks so it can reach
    >75% top-1 accuracy after full training, while remaining compact enough for
    subsequent pruning experiments.
    """

    def residual_block(
        x: tf.Tensor,
        filters: int,
        downsample: bool = False,
        name: Optional[str] = None,
    ) -> tf.Tensor:
        stride = 2 if downsample else 1
        shortcut = x

        y = tf.keras.layers.Conv2D(
            filters,
            kernel_size=3,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            name=None if name is None else f"{name}_conv1",
        )(x)
        y = tf.keras.layers.BatchNormalization(name=None if name is None else f"{name}_bn1")(y)
        y = tf.keras.layers.Activation("relu")(y)
        y = tf.keras.layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            name=None if name is None else f"{name}_conv2",
        )(y)
        y = tf.keras.layers.BatchNormalization(name=None if name is None else f"{name}_bn2")(y)

        if downsample or shortcut.shape[-1] != filters:
            shortcut = tf.keras.layers.Conv2D(
                filters,
                kernel_size=1,
                strides=stride,
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal",
                name=None if name is None else f"{name}_proj",
            )(shortcut)
            shortcut = tf.keras.layers.BatchNormalization(
                name=None if name is None else f"{name}_proj_bn"
            )(shortcut)

        out = tf.keras.layers.Add()([shortcut, y])
        out = tf.keras.layers.Activation("relu")(out)
        return out

    def scaled_filters(filters: int) -> int:
        return max(16, int(filters * width_multiplier))

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    x = tf.keras.layers.Conv2D(
        scaled_filters(64),
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = residual_block(x, scaled_filters(64), name="block1a")
    x = residual_block(x, scaled_filters(64), name="block1b")
    x = residual_block(x, scaled_filters(128), downsample=True, name="block2a")
    x = residual_block(x, scaled_filters(128), name="block2b")
    x = residual_block(x, scaled_filters(256), downsample=True, name="block3a")
    x = residual_block(x, scaled_filters(256), name="block3b")
    x = residual_block(x, scaled_filters(512), downsample=True, name="block4a")
    x = residual_block(x, scaled_filters(512), name="block4b")

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        kernel_initializer="he_normal",
    )(x)

    model = tf.keras.Model(inputs, outputs, name="cifar100_resnet")
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    # Compile with an adaptive loss/metric stack so evaluations tolerate either
    # sparse integer labels or one-hot vectors while preserving label smoothing.
    model.compile(
        optimizer=optimizer,
        loss=AdaptiveCategoricalCrossentropy(num_classes=num_classes, label_smoothing=0.05),
        metrics=[
            AdaptiveCategoricalAccuracy(num_classes=num_classes, name="accuracy"),
            AdaptiveTopKCategoricalAccuracy(num_classes=num_classes, k=5, name="top5"),
        ],
    )
    return model


def prepare_compression_datasets(
    val_size: int = VAL_SIZE,
    calibration_size: int = CALIBRATION_SIZE,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Load and normalize CIFAR-100, returning numpy arrays for downstream tasks.

    Returns:
        tuple: (x_train, y_train, x_val, y_val, x_test, y_test, calibration_data)
    """

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    mean = np.mean(x_train, axis=(0, 1, 2), keepdims=True)
    std = np.std(x_train, axis=(0, 1, 2), keepdims=True) + 1e-7
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x_train))
    x_train = x_train[indices]
    y_train = y_train[indices]

    if val_size <= 0 or val_size >= len(x_train):
        raise ValueError("val_size must be between 1 and len(x_train) - 1")
    x_val = x_train[:val_size]
    y_val = y_train[:val_size]
    x_train = x_train[val_size:]
    y_train = y_train[val_size:]

    if calibration_size >= len(x_train):
        raise ValueError("calibration_size must be smaller than remaining training set")
    calibration_indices = rng.choice(len(x_train), size=calibration_size, replace=False)
    calibration_data = (x_train[calibration_indices], y_train[calibration_indices])

    return x_train, y_train, x_val, y_val, x_test, y_test, calibration_data


def train_baseline_model(
    epochs: int = 30,
    batch_size: int = 128,
    output_path: str = DEFAULT_MODEL_PATH,
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    seed: int = 42,
) -> Dict:
    """
    Train the CIFAR-100 baseline model and export it for pruning stages.

    Returns:
        dict: training history, final accuracy, and saved model paths.
    """

    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        calibration_data,
    ) = prepare_compression_datasets(seed=seed)

    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Instantiate model first so we can perform dataset label conversion
    # (one-hot) consistent with `CategoricalCrossentropy` which expects
    # categorical labels.
    model = create_baseline_model()

    train_ds = _build_dataset(
        x_train, y_train, batch_size, augment=True, shuffle=True, num_classes=model.output_shape[-1]
    )
    val_ds = _build_dataset(x_val, y_val, batch_size, num_classes=model.output_shape[-1])
    test_ds = _build_dataset(x_test, y_test, batch_size, num_classes=model.output_shape[-1])

    Path(os.path.dirname(checkpoint_path) or ".").mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=2,
    )

    if os.path.exists(checkpoint_path):
        model = tf.keras.models.load_model(checkpoint_path, custom_objects=CUSTOM_OBJECTS)

    test_accuracy = model.evaluate(test_ds, verbose=0)[1]
    model.save(output_path)

    metadata = {
        "model_path": output_path,
        "checkpoint_path": checkpoint_path,
        "test_accuracy": float(test_accuracy),
        "history": history.history,
        "calibration_size": len(calibration_data[0]),
    }
    return metadata


# --------------------------------------------------------------------------- #
# Internal utilities
# --------------------------------------------------------------------------- #
def _build_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    augment: bool = False,
    shuffle: bool = False,
    num_classes: Optional[int] = None,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    # If `num_classes` is set, convert sparse int labels to one-hot vectors so
    # they work with `CategoricalCrossentropy` and categorical metrics.
    if isinstance(num_classes, int):
        def _to_one_hot(x, y):
            # y may be shape (,) or (N,1) - ensure a 1D int tensor
            y = tf.convert_to_tensor(y)
            if len(y.shape) > 0 and y.shape[-1] == 1:
                y = tf.squeeze(y, axis=-1)
            return x, tf.one_hot(y, depth=num_classes)

        ds = ds.map(_to_one_hot, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(features), seed=1337, reshuffle_each_iteration=True)
    if augment:
        augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomTranslation(0.1, 0.1),
                tf.keras.layers.RandomRotation(0.05),
                tf.keras.layers.RandomZoom((-0.1, 0.1), (-0.1, 0.1)),
                tf.keras.layers.RandomContrast(0.1),
            ],
            name="baseline_augmentation",
        )

        def apply_aug(x, y):
            return augmentation(x, training=True), y

        ds = ds.map(apply_aug, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)
