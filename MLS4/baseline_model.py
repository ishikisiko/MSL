from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import tf_compat  # noqa: F401  # keep tfmot compatible with newer TensorFlow builds
import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE
CALIBRATION_SIZE = 1000
VAL_SIZE = 5000
DEFAULT_MODEL_PATH = "baseline_model.keras"
DEFAULT_CHECKPOINT_PATH = "checkpoints/baseline_best.keras"
RESULTS_DIR = Path("results")
REPORTS_DIR = Path("reports")
DATASET_CACHE_PATH = RESULTS_DIR / "datasets" / "cifar100_splits_v1.npz"
CALIBRATION_EXPORT_PATH = RESULTS_DIR / "datasets" / "calibration_samples.npz"
TRAIN_HISTORY_PATH = RESULTS_DIR / "baseline_training_history.json"
BASELINE_REPORT_PATH = REPORTS_DIR / "baseline_summary.json"
DROP_REMAINDER_BATCH_SIZE = 256
DEFAULT_BATCHNORM_PASSES = 200


RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
CALIBRATION_EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


_AUGMENTATION_PIPELINE = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="reflect"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ],
    name="cifar100_train_aug",
)


def _augment_batch(images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    return _AUGMENTATION_PIPELINE(images, training=True), labels


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

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> None:
        y_true = _ensure_one_hot_labels(y_true, self.num_classes, y_pred.dtype)
        return super().update_state(y_true, y_pred, sample_weight=sample_weight)

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

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> None:
        y_true = _ensure_one_hot_labels(y_true, self.num_classes, y_pred.dtype)
        return super().update_state(y_true, y_pred, sample_weight=sample_weight)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config


@tf.keras.utils.register_keras_serializable(package="mls4")
class LearningRateLogger(tf.keras.callbacks.Callback):
    """Callback to log learning rate at the end of each epoch and batch."""
    
    def __init__(self, log_freq: str = "epoch", verbose: int = 1):
        """
        Args:
            log_freq: Frequency of logging, either 'epoch' or 'batch'
            verbose: Verbosity mode (0=silent, 1=print, 2=print with more details)
        """
        super().__init__()
        self.log_freq = log_freq
        self.verbose = verbose
        
    def on_epoch_end(self, epoch, logs=None):
        if self.log_freq == "epoch":
            lr = self.model.optimizer.lr
            if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                # For learning rate schedules, we need to compute the current value
                current_step = self.model.optimizer.iterations
                lr = lr(current_step)
                lr = float(tf.keras.backend.get_value(lr))
            else:
                lr = float(tf.keras.backend.get_value(lr))
                
            if self.verbose >= 1:
                print(f"\nEpoch {epoch + 1}: Learning Rate = {lr:.6f}")
                
            if logs is None:
                logs = {}
            logs['lr'] = lr
    
    def on_train_batch_end(self, batch, logs=None):
        if self.log_freq == "batch":
            lr = self.model.optimizer.lr
            if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                # For learning rate schedules, we need to compute the current value
                current_step = self.model.optimizer.iterations
                lr = lr(current_step)
                lr = float(tf.keras.backend.get_value(lr))
            else:
                lr = float(tf.keras.backend.get_value(lr))
                
            if self.verbose >= 2 and batch % 100 == 0:  # Print every 100 batches to avoid spam
                print(f"Batch {batch}: Learning Rate = {lr:.6f}")
                
            if logs is None:
                logs = {}
            logs['lr'] = lr

    def get_config(self) -> Dict[str, Any]:
        """Returns the serializable config of the callback."""
        config = super().get_config()
        config.update({
            "log_freq": self.log_freq,
            "verbose": self.verbose,
        })
        return config


CUSTOM_OBJECTS: Dict[str, Any] = {
    "AdaptiveCategoricalCrossentropy": AdaptiveCategoricalCrossentropy,
    "AdaptiveCategoricalAccuracy": AdaptiveCategoricalAccuracy,
    "AdaptiveTopKCategoricalAccuracy": AdaptiveTopKCategoricalAccuracy,
    "LearningRateLogger": LearningRateLogger,
}


def build_optimizer(
    optimizer_name: str,
    learning_rate: Union[float, tf.keras.optimizers.schedules.LearningRateSchedule],
    weight_decay: float,
    momentum: float = 0.9,
) -> tf.keras.optimizers.Optimizer:
    name = optimizer_name.lower()
    if name == "adamw":
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        )
    if name == "sgdw":
        return tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=True,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Use 'adamw' or 'sgdw'.")


def create_baseline_model(
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 100,
    dropout_rate: float = 0.3,
    width_multiplier: float = 1.0,
    optimizer_name: str = "adamw",
    learning_rate: Union[float, tf.keras.optimizers.schedules.LearningRateSchedule] = 5e-4,
    weight_decay: float = 5e-5,
) -> tf.keras.Model:
    """Create an EfficientNet-B0 based baseline tailored for CIFAR-100."""

    resize_target = 128
    bottleneck_units = max(128, int(1280 * width_multiplier))

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Resizing(resize_target, resize_target, name="resize_to_128")(inputs)
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,
        input_tensor=x,
        drop_connect_rate=0.2,
    )
    x = backbone.output
    x = tf.keras.layers.BatchNormalization(name="backbone_bn")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="pre_projection_dropout")(x)
    x = tf.keras.layers.Dense(
        bottleneck_units,
        activation="swish",
        kernel_initializer="he_normal",
        name="projection_dense",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="projection_bn")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="post_projection_dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        kernel_initializer="lecun_normal",
        name="classification_head",
    )(x)

    model = tf.keras.Model(inputs, outputs, name="cifar100_efficientnet_b0")
    optimizer = build_optimizer(
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    model.compile(
        optimizer=optimizer,
        loss=AdaptiveCategoricalCrossentropy(num_classes=num_classes, label_smoothing=0.1),
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
    cache_path: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Load CIFAR-100 with deterministic splits and persistent caching."""

    cache_file = Path(cache_path) if cache_path is not None else DATASET_CACHE_PATH
    cache_payload = _load_cached_dataset(cache_file)
    if cache_payload:
        meta = cache_payload["metadata"]
        if (
            meta.get("version", 0) == 1
            and meta["val_size"] == val_size
            and meta["calibration_size"] == calibration_size
            and meta["seed"] == seed
        ):
            arrays = cache_payload["arrays"]
        else:
            cache_payload = None
    if cache_payload is None:
        (raw_train, raw_train_labels), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        raw_train = raw_train.astype("float32")
        x_test = x_test.astype("float32")

        mean = np.mean(raw_train, axis=(0, 1, 2), keepdims=True)
        std = np.std(raw_train, axis=(0, 1, 2), keepdims=True) + 1e-7
        raw_train = (raw_train - mean) / std
        x_test = (x_test - mean) / std

        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(raw_train))
        raw_train = raw_train[indices]
        raw_train_labels = raw_train_labels[indices]

        if val_size <= 0 or val_size >= len(raw_train):
            raise ValueError("val_size must be between 1 and len(x_train) - 1")
        x_val = raw_train[:val_size]
        y_val = raw_train_labels[:val_size]
        x_train = raw_train[val_size:]
        y_train = raw_train_labels[val_size:]

        if calibration_size >= len(x_train):
            raise ValueError("calibration_size must be smaller than remaining training set")
        calibration_indices = rng.choice(len(x_train), size=calibration_size, replace=False)
        calibration_data = (x_train[calibration_indices], y_train[calibration_indices])

        arrays = {
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "x_test": x_test,
            "y_test": y_test,
            "calibration_x": calibration_data[0],
            "calibration_y": calibration_data[1],
        }
        metadata = {
            "seed": seed,
            "val_size": val_size,
            "calibration_size": calibration_size,
            "mean": mean.mean().item(),
            "std": std.mean().item(),
            "version": 1,
        }
        _persist_dataset_cache(cache_file, arrays, metadata)
    else:
        arrays = cache_payload["arrays"]

    calibration_data = (arrays["calibration_x"], arrays["calibration_y"])
    _export_calibration_samples(calibration_data)

    return (
        arrays["x_train"],
        arrays["y_train"],
        arrays["x_val"],
        arrays["y_val"],
        arrays["x_test"],
        arrays["y_test"],
        calibration_data,
    )


def train_baseline_model(
    epochs: int = 100,
    batch_size: int = DROP_REMAINDER_BATCH_SIZE,
    output_path: str = DEFAULT_MODEL_PATH,
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    seed: int = 42,
    optimizer_name: str = "adamw",
    base_learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    ema_decay: Optional[float] = 0.999,
) -> Dict[str, Any]:
    """Train the EfficientNet-B0 baseline with modern regularization."""

    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        calibration_data,
    ) = prepare_compression_datasets(seed=seed)

    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:  # pragma: no cover - fallback for older TF builds
        tf.random.set_seed(seed)
        np.random.seed(seed)
    Path(os.path.dirname(checkpoint_path) or ".").mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)

    steps_per_epoch = math.ceil(len(x_train) / batch_size)
    total_steps = steps_per_epoch * max(epochs, 1)
    first_decay_steps = max(total_steps // 5, steps_per_epoch, 1)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=base_learning_rate,
        first_decay_steps=first_decay_steps,
        t_mul=1.5,
        m_mul=0.9,
        alpha=0.02,
    )

    model = create_baseline_model(
        optimizer_name=optimizer_name,
        learning_rate=lr_schedule,
        weight_decay=weight_decay,
    )
    num_classes = model.output_shape[-1]

    train_ds = _build_dataset(
        x_train,
        y_train,
        batch_size,
        augment=True,
        shuffle=True,
        num_classes=num_classes,
        drop_remainder=True,
    )
    val_ds = _build_dataset(
        x_val,
        y_val,
        batch_size,
        augment=False,
        shuffle=False,
        num_classes=num_classes,
    )
    test_ds = _build_dataset(
        x_test,
        y_test,
        batch_size,
        augment=False,
        shuffle=False,
        num_classes=num_classes,
    )
    bn_ds = _build_dataset(
        x_train,
        y_train,
        batch_size,
        augment=False,
        shuffle=False,
        num_classes=num_classes,
    )

    csv_logger = tf.keras.callbacks.CSVLogger(str(RESULTS_DIR / "baseline_training_log.csv"), append=False)
    # Add learning rate logger to track current learning rate
    lr_logger = LearningRateLogger(log_freq="epoch", verbose=1)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
        csv_logger,
        lr_logger,
    ]
    ema_callback = None
    if ema_decay is not None and ema_decay < 1.0:
        ema_cls = getattr(tf.keras.callbacks, "ExponentialMovingAverage", None)
        if ema_cls is None:
            experimental_api = getattr(tf.keras.callbacks, "experimental", None)
            if experimental_api is not None:
                ema_cls = getattr(experimental_api, "ExponentialMovingAverage", None)
        if ema_cls is not None:
            ema_callback = ema_cls(decay=ema_decay, update_freq="batch")
            callbacks.append(ema_callback)

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=2,
    )

    if ema_callback is not None:
        ema_callback.assign_average_vars(model.variables)

    _recompute_batchnorm_statistics(model, bn_ds, max_batches=DEFAULT_BATCHNORM_PASSES)

    test_metrics = model.evaluate(test_ds, verbose=0)
    test_accuracy = float(test_metrics[1]) if isinstance(test_metrics, (list, tuple)) else float(test_metrics)
    model.save(output_path)

    history_payload = _serialize_history(history.history)
    TRAIN_HISTORY_PATH.write_text(json.dumps(history_payload, indent=2))

    summary = {
        "model_path": output_path,
        "checkpoint_path": checkpoint_path,
        "test_accuracy": test_accuracy,
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": model.optimizer.get_config(),
        "optimizer_name": optimizer_name,
        "weight_decay": weight_decay,
        "ema_decay": ema_decay,
        "dataset_cache": str(DATASET_CACHE_PATH),
        "calibration_file": str(CALIBRATION_EXPORT_PATH),
        "history_file": str(TRAIN_HISTORY_PATH),
        "calibration_size": int(len(calibration_data[0])),
    }
    BASELINE_REPORT_PATH.write_text(json.dumps(summary, indent=2))

    metadata = {
        **summary,
        "history": history_payload,
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
    drop_remainder: bool = False,
) -> tf.data.Dataset:
    features = np.asarray(features, dtype=np.float32)
    labels_array = np.asarray(labels)
    if isinstance(num_classes, int):
        labels_array = tf.keras.utils.to_categorical(labels_array.reshape(-1), num_classes)
        labels_array = labels_array.astype("float32")
    ds = tf.data.Dataset.from_tensor_slices((features, labels_array))
    if shuffle:
        buffer = min(len(features), 10000)
        ds = ds.shuffle(buffer_size=buffer, seed=1337, reshuffle_each_iteration=True)
    if augment:
        ds = ds.map(_augment_batch, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size, drop_remainder=drop_remainder).prefetch(AUTOTUNE)


def _persist_dataset_cache(
    cache_path: Path,
    arrays: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
) -> None:
    payload = {key: value for key, value in arrays.items()}
    payload["metadata"] = np.array(json.dumps(metadata))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)


def _load_cached_dataset(cache_path: Path) -> Optional[Dict[str, Any]]:
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            metadata_raw = data["metadata"].item()
            arrays = {key: data[key] for key in data.files if key != "metadata"}
        metadata = json.loads(metadata_raw)
    except Exception:
        return None

    for key, value in list(arrays.items()):
        if key.startswith("x_") or key.startswith("calibration_x"):
            arrays[key] = value.astype("float32")
        else:
            arrays[key] = value.astype("int32")
    return {"arrays": arrays, "metadata": metadata}


def _export_calibration_samples(calibration_data: Tuple[np.ndarray, np.ndarray]) -> Path:
    CALIBRATION_EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CALIBRATION_EXPORT_PATH,
        x=calibration_data[0],
        y=calibration_data[1],
    )
    return CALIBRATION_EXPORT_PATH


def _recompute_batchnorm_statistics(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    max_batches: Optional[int] = None,
) -> None:
    bn_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]
    if not bn_layers:
        return
    for step, (images, _) in enumerate(dataset):
        model(images, training=True)
        if max_batches is not None and step + 1 >= max_batches:
            break


def _serialize_history(history: Dict[str, Any]) -> Dict[str, Any]:
    serialized: Dict[str, Any] = {}
    for key, values in history.items():
        serialized[key] = [float(v) for v in values]
    return serialized
