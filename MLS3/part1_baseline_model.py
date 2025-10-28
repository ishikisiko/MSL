"""Baseline MobileNetV2 implementation for the MLS3 assignment.

REFACTORED (V5):
- Restored in-model resizing so CIFAR-10 inputs stay lightweight in the tf.data pipeline.
- Enabled tf.data caching and GPU prefetch to better saturate single-GPU pipelines.
- Added automatic distribution strategy selection to exploit multi-GPU setups.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import psutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2_as_graph,
)

NATIVE_INPUT_SHAPE = (32, 32, 3)
TARGET_INPUT_SHAPE = (96, 96, 3)
NUM_CLASSES = 10
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.1

try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as exc:
    print(f"Error initializing GPU: {exc}")

LOGICAL_GPUS = tf.config.list_logical_devices("GPU")
AUTOTUNE = tf.data.AUTOTUNE


def print_device_info() -> None:
    """打印详细的可用计算设备信息。"""
    print("\n" + "=" * 70)
    print("计算设备信息")
    print("=" * 70)
    print(f"TensorFlow 版本: {tf.__version__}")
    physical_gpus = tf.config.list_physical_devices("GPU")
    if physical_gpus:
        print(f"\n✓ 检测到 {len(physical_gpus)} 张 GPU - 训练将使用 GPU 加速")
    else:
        print("\n✗ 未检测到 GPU - 训练将使用 CPU")
    memory = psutil.virtual_memory()
    print(f"\n系统内存 (RAM): {memory.total / (1024**3):.2f} GB")
    print(f"可用内存 (RAM): {memory.available / (1024**3):.2f} GB")
    print("=" * 70 + "\n")


def _select_distribution_strategy() -> tf.distribute.Strategy:
    """选择合适的分布式策略以提升并行效率。"""
    gpu_count = len(LOGICAL_GPUS)
    if gpu_count > 1:
        print(
            f"\nDetected {gpu_count} GPUs - enabling MirroredStrategy for data parallel training."
        )
        return tf.distribute.MirroredStrategy()
    if gpu_count == 1:
        print("\nDetected single GPU - using default strategy with GPU prefetch.")
    else:
        print("\nNo GPU detected - using CPU execution.")
    return tf.distribute.get_strategy()


def _build_data_augmentation() -> keras.Sequential:
    """创建数据增强流程（在模型内部使用）。"""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            layers.RandomZoom(0.2),
        ],
        name="data_augmentation",
    )


def create_baseline_model(
    input_shape: Tuple[int, int, int] = NATIVE_INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
) -> keras.Model:
    """创建一个基于 MobileNetV2 的分类器。"""

    inputs = keras.Input(shape=input_shape, name="input_image")
    x = layers.Resizing(
        TARGET_INPUT_SHAPE[0],
        TARGET_INPUT_SHAPE[1],
        name="resize_to_backbone",
    )(inputs)
    x = _build_data_augmentation()(x)

    backbone = keras.applications.MobileNetV2(
        input_shape=TARGET_INPUT_SHAPE,
        include_top=False,
        weights="imagenet",
        pooling=None,
        name="mobilenetv2_backbone",
    )
    backbone.trainable = False
    x = backbone(x, training=False)

    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dropout(0.2, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="baseline_mobilenetv2")


def load_and_preprocess_data(
    batch_size: int = BATCH_SIZE,
    validation_split: float = VALIDATION_SPLIT,
    cache_dataset: bool = True,
    enable_gpu_prefetch: bool = True,
    drop_remainder: bool = False,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """加载 CIFAR-10 并构建高吞吐量输入管线。"""

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    val_size = int(len(x_train) * validation_split)
    if val_size == 0:
        raise ValueError("validation_split 太小。")

    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]

    print(f"训练样本数: {len(x_train)}")
    print(f"验证样本数: {len(x_val)}")
    print(f"测试样本数: {len(x_test)}")

    target_device: Optional[str] = None
    if enable_gpu_prefetch and len(LOGICAL_GPUS) == 1:
        target_device = "/GPU:0"

    def _prepare_dataset(
        images: np.ndarray, labels: np.ndarray, shuffle: bool
    ) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        options = tf.data.Options()
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.autotune = True
        options.experimental_slack = True
        options.experimental_deterministic = not shuffle
        ds = ds.with_options(options)

        if cache_dataset:
            ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(buffer_size=len(images), reshuffle_each_iteration=True)

        def _preprocess(img, lbl):
            img = tf.cast(img, tf.float32)
            img = mobilenet_v2.preprocess_input(img)
            lbl = tf.squeeze(lbl, axis=-1)
            return img, lbl

        ds = ds.map(_preprocess, num_parallel_calls=AUTOTUNE)
        batch_drop = drop_remainder if shuffle else False
        ds = ds.batch(batch_size, drop_remainder=batch_drop)

        if target_device is not None:
            ds = ds.apply(tf.data.experimental.copy_to_device(target_device))

        ds = ds.prefetch(AUTOTUNE)
        return ds

    train_ds = _prepare_dataset(x_train, y_train, shuffle=True)
    val_ds = _prepare_dataset(x_val, y_val, shuffle=False)
    test_ds = _prepare_dataset(x_test, y_test, shuffle=False)

    return train_ds, val_ds, test_ds


def _calculate_model_size_mb(model: keras.Model) -> float:
    size_bytes = 0
    for weight in model.weights:
        dtype = np.dtype(str(weight.dtype))
        size_bytes += np.prod(weight.shape) * dtype.itemsize
    return size_bytes / (1024**2)


def _calculate_model_flops(model: keras.Model, batch_size: int = 1) -> int:
    try:
        inputs = tf.TensorSpec([batch_size, *model.input_shape[1:]], tf.float32)
        concrete_fn = tf.function(model).get_concrete_function(inputs)
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_fn)
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name="")
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(
                graph=graph, run_meta=run_meta, cmd="op", options=opts
            )
        return flops.total_float_ops if flops is not None else 0
    except Exception:
        return 0


def benchmark_baseline_model(
    model: keras.Model, test_data: Iterable, batch_size: int = BATCH_SIZE
) -> Dict[str, float]:
    dataset = test_data.prefetch(AUTOTUNE)
    process = psutil.Process(os.getpid())
    sample_batch = next(iter(dataset))
    sample_inputs = sample_batch[0][:1]
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
        "memory_usage_mb": float(max(after_mem - before_mem, 0) / (1024**2)),
        "model_size_mb": float(_calculate_model_size_mb(model)),
        "accuracy": accuracy,
        "flops": float(_calculate_model_flops(model, batch_size=1)),
        "parameters": float(model.count_params()),
    }
    metrics.update({f"eval_{k}": float(v) for k, v in evaluation.items()})
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline MobileNetV2 training/evaluation")
    parser.add_argument("--eval-only", action="store_true", help="仅评估：跳过训练，直接加载模型并评估")
    parser.add_argument(
        "--model-path",
        type=str,
        default="baseline_mobilenetv2.keras",
        help="评估/保存的模型路径",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="训练/评估的 batch size"
    )
    args = parser.parse_args()

    print_device_info()

    strategy = _select_distribution_strategy()
    replicas = getattr(strategy, "num_replicas_in_sync", 1)
    if replicas > 1 and args.batch_size % replicas != 0:
        print(
            f"\nWarning: batch_size={args.batch_size} is not divisible by {replicas} replicas; "
            "consider adjusting it for balanced GPU load."
        )

    print("正在加载和准备数据集 (32x32 -> 96x96)...")
    train_ds, val_ds, test_ds = load_and_preprocess_data(
        batch_size=args.batch_size,
        validation_split=VALIDATION_SPLIT,
        cache_dataset=True,
        enable_gpu_prefetch=(len(LOGICAL_GPUS) == 1),
        drop_remainder=replicas > 1,
    )

    if args.eval_only:
        model_file = Path(args.model_path)
        if not model_file.exists():
            print(
                f"\n[eval-only] 指定的模型文件不存在: {model_file.resolve()}\n将继续执行训练流程…"
            )
        else:
            print(f"\n=== Eval-only: 从 '{model_file}' 加载模型并在测试集上评估 ===")
            model = keras.models.load_model(str(model_file))
            metrics = benchmark_baseline_model(model, test_ds, batch_size=args.batch_size)

            print("\nBaseline Model Performance:")
            print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Test Loss: {metrics.get('eval_loss', float('nan')):.4f}")
            if "single_inference_time" in metrics:
                print(
                    f"  Single Inference Time (s): {metrics['single_inference_time']:.6f}"
                )
            if "batch_inference_time" in metrics:
                print(
                    f"  Batch Inference Time (s): {metrics['batch_inference_time']:.6f}"
                )
            if "memory_usage_mb" in metrics:
                print(
                    f"  Inference Memory Delta (MB): {metrics['memory_usage_mb']:.2f}"
                )
            print(f"  Total Parameters: {metrics['parameters']:.0f}")
            print(f"  Model Size (MB): {metrics['model_size_mb']:.2f}")
            if "flops" in metrics:
                print(f"  Estimated FLOPs: {int(metrics['flops'])}")
            exit(0)

    with strategy.scope():
        print("\n=== 阶段 1: 训练分类头 (Backbone 冻结) ===")
        print("CPU 管道负责归一化与缓存，GPU 负责上采样与数据增强。")
        model = create_baseline_model()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        print("模型摘要 (阶段 1 - Backbone 冻结):")
        model.summary()

        callbacks_phase1 = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=15,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                "baseline_mobilenetv2_phase1.keras",
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
        ]

        history_phase1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=50,
            callbacks=callbacks_phase1,
            verbose=1,
        )

        print("\n=== Phase 2: Fine-tuning (unfreezing last 50 layers) ===")
        backbone = model.get_layer("mobilenetv2_backbone")
        backbone.trainable = True

        FINE_TUNE_AT = -50
        for layer in backbone.layers[:FINE_TUNE_AT]:
            layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        print("模型摘要 (阶段 2 - 微调):")
        model.summary()

        callbacks_phase2 = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=15,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                "baseline_mobilenetv2_final.keras",
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
        ]

        history_phase2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=30,
            callbacks=callbacks_phase2,
            verbose=1,
        )

        print("\n=== Saving final model ===")
        model.save(args.model_path)

        print("\n=== Evaluating final model on test set ===")
        metrics = benchmark_baseline_model(model, test_ds, batch_size=args.batch_size)

    print("\nBaseline Model Performance:")
    print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Test Loss: {metrics['eval_loss']:.4f}")
    if "single_inference_time" in metrics:
        print(f"  Single Inference Time (s): {metrics['single_inference_time']:.6f}")
    if "batch_inference_time" in metrics:
        print(f"  Batch Inference Time (s): {metrics['batch_inference_time']:.6f}")
    if "memory_usage_mb" in metrics:
        print(f"  Inference Memory Delta (MB): {metrics['memory_usage_mb']:.2f}")
    print(f"  Total Parameters: {metrics['parameters']:.0f}")
    print(f"  Model Size (MB): {metrics['model_size_mb']:.2f}")
    if "flops" in metrics:
        print(f"  Estimated FLOPs: {int(metrics['flops'])}")

    print("\n=== Training Summary ===")
    print(f"Phase 1 - Best validation accuracy: {max(history_phase1.history['val_accuracy']):.4f}")
    val_acc_phase2 = history_phase2.history.get("val_accuracy", [0])
    print(f"Phase 2 - Best validation accuracy: {max(val_acc_phase2): .4f}")
    print(f"Final test accuracy: {metrics['accuracy']:.4f}")

    if metrics['accuracy'] >= 0.85:
        print(
            f"\nSUCCESS: Model achieved {metrics['accuracy'] * 100:.2f}% accuracy (target: 85%)"
        )
    else:
        print(
            f"\nModel accuracy {metrics['accuracy'] * 100:.2f}% is below target (85%)"
        )