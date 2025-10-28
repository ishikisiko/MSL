"""Baseline MobileNetV2 implementation for the MLS3 assignment.

REFACTORED (V4):
- FUNDAMENTAL FIX: The model was failing because the 32x32 input shape
  invalidated the ImageNet pre-trained weights.
- Added an `UpSampling2D` layer to scale CIFAR-10's (32, 32, 3) input
  to (96, 96, 3) *inside* the model.
- The MobileNetV2 backbone is now correctly instantiated with
  `input_shape=(96, 96, 3)` to properly load and leverage the
  pre-trained weights.
- All other logic (Phase 1 optimizer, data loading) remains stable.
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
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

# --- 全局常量 ---
# 模型接收的原始输入尺寸
NATIVE_INPUT_SHAPE = (32, 32, 3)
# Backbone 期望的、用于迁移学习的尺寸
TARGET_INPUT_SHAPE = (96, 96, 3) 
NUM_CLASSES = 10
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.1

# ... (GPU 设置 & AUTOTUNE 保持不变) ...
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(f"Error initializing GPU: {e}")
AUTOTUNE = tf.data.AUTOTUNE


def print_device_info() -> None:
    """打印详细的可用计算设备信息。"""
    print("\n" + "="*70)
    print("计算设备信息")
    print("="*70)
    print(f"TensorFlow 版本: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n✓ 检测到 GPU - 训练将使用 GPU 加速")
    else:
        print("\n✗ 未检测到 GPU - 训练将使用 CPU")
    memory = psutil.virtual_memory()
    print(f"\n系统内存 (RAM): {memory.total / (1024**3):.2f} GB")
    print(f"可用内存 (RAM): {memory.available / (1024**3):.2f} GB")
    print("="*70 + "\n")


def _build_data_augmentation() -> keras.Sequential:
    """创建数据增强流程（在模型内部使用）。"""
    # 这些增强现在将在 96x96 图像上操作
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            layers.RandomZoom(0.2), # 在 96x96 上可以安全使用 Zoom
        ],
        name="data_augmentation",
    )


def create_baseline_model(
    input_shape: Tuple[int, int, int] = TARGET_INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
) -> keras.Model:
    """
    创建一个基于 MobileNetV2 的分类器。
    
    模型现在直接接收 (96, 96, 3) 的输入，预处理已在数据管道中完成。
    """

    inputs = keras.Input(shape=input_shape, name="input_image")
    
    # 预处理和数据增强已移至 tf.data 管道
    # x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale_inputs")(inputs)

    # Backbone (主干网络)
    # 关键修复：Backbone 必须用 TARET_SHAPE 实例化！
    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling=None,
        name="mobilenetv2_backbone"
    )
    
    # 默认冻结 backbone (用于第一阶段)
    backbone.trainable = False

    # 以推理模式 (training=False) 调用冻结的 backbone
    x = backbone(inputs, training=False) 

    # 5. 分类头 (轻量级)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dropout(0.2, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="baseline_mobilenetv2")
    
    return model


def load_and_preprocess_data(
    batch_size: int = BATCH_SIZE,
    validation_split: float = VALIDATION_SPLIT,
    target_shape: Tuple[int, int, int] = TARGET_INPUT_SHAPE,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    加载 CIFAR-10 并应用预处理。
    - 上采样、数据增强和归一化现在都在 tf.data 管道中完成。
    """

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

    # 数据增强层现在用于 tf.data.Dataset.map()
    data_augmentation = _build_data_augmentation()
    upsample_factor = target_shape[0] // NATIVE_INPUT_SHAPE[0]

    def _prepare_dataset(images: np.ndarray, labels: np.ndarray, shuffle: bool, augment: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(images), reshuffle_each_iteration=True)
        
        # 将所有预处理操作移到这里
        def _preprocess(img, lbl):
            img = tf.cast(img, tf.float32)
            # 1. 上采样
            if upsample_factor > 1:
                img = tf.image.resize(img, (target_shape[0], target_shape[1]), method='bilinear')
            # 2. 数据增强 (仅用于训练集)
            if augment:
                img = data_augmentation(img, training=True)
            # 3. 归一化
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
            lbl = tf.squeeze(lbl, axis=-1)
            return img, lbl

        ds = ds.map(_preprocess, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
        return ds

    train_ds = _prepare_dataset(x_train, y_train, shuffle=True, augment=True)
    val_ds = _prepare_dataset(x_val, y_val, shuffle=False, augment=False)
    test_ds = _prepare_dataset(x_test, y_test, shuffle=False, augment=False)

    return train_ds, val_ds, test_ds


# --- 基准测试函数 (未修改) ---
def _calculate_model_size_mb(model: keras.Model) -> float:
    size_bytes = 0
    for weight in model.weights:
        dtype = np.dtype(str(weight.dtype))
        size_bytes += np.prod(weight.shape) * dtype.itemsize
    return size_bytes / (1024 ** 2)

def _calculate_model_flops(model: keras.Model, batch_size: int = 1) -> int:
    try:
        # 注意: FLOPs 计算现在基于 NATIVE_INPUT_SHAPE
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
    # CLI 选项：支持仅评估模式，直接加载已保存模型并输出完整指标
    parser = argparse.ArgumentParser(description="Baseline MobileNetV2 training/evaluation")
    parser.add_argument("--eval-only", action="store_true", help="仅评估：跳过训练，直接加载模型并评估")
    parser.add_argument("--model-path", type=str, default="baseline_mobilenetv2.keras", help="评估/保存的模型路径")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="训练/评估的 batch size")
    args = parser.parse_args()

    print_device_info()
    
    device_name = '/GPU:0' if gpus else '/CPU:0'
    with tf.device(device_name):
        print("正在加载和准备数据集 (32x32)...")
        train_ds, val_ds, test_ds = load_and_preprocess_data(
            batch_size=args.batch_size, validation_split=VALIDATION_SPLIT
        )

        # 仅评估路径：如果提供 --eval-only 并且模型文件存在，则直接加载并基准测试
        if args.eval_only:
            model_file = Path(args.model_path)
            if not model_file.exists():
                print(f"\n[eval-only] 指定的模型文件不存在: {model_file.resolve()}\n将继续执行训练流程…")
            else:
                print(f"\n=== Eval-only: 从 '{model_file}' 加载模型并在测试集上评估 ===")
                model = keras.models.load_model(str(model_file))
                metrics = benchmark_baseline_model(model, test_ds, batch_size=args.batch_size)

                print("\nBaseline Model Performance:")
                print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Test Loss: {metrics.get('eval_loss', float('nan')):.4f}")
                if 'single_inference_time' in metrics:
                    print(f"  Single Inference Time (s): {metrics['single_inference_time']:.6f}")
                if 'batch_inference_time' in metrics:
                    print(f"  Batch Inference Time (s):  {metrics['batch_inference_time']:.6f}")
                if 'memory_usage_mb' in metrics:
                    print(f"  Inference Memory Delta (MB): {metrics['memory_usage_mb']:.2f}")
                print(f"  Total Parameters: {metrics['parameters']:.0f}")
                print(f"  Model Size (MB): {metrics['model_size_mb']:.2f}")
                if 'flops' in metrics:
                    print(f"  Estimated FLOPs: {int(metrics['flops'])}")

                # 在 eval-only 模式下无需训练总结
                exit(0)
        
        print("\n=== 阶段 1: 训练分类头 (Backbone 冻结) ===")
        print("数据管道将 (32x32) -> (96x96) 以适配预训练权重。")
        model = create_baseline_model(
            input_shape=TARGET_INPUT_SHAPE,
            num_classes=NUM_CLASSES
        )
        
        # 使用 V3 验证过的稳定优化器策略
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        print("模型摘要 (阶段 1 - Backbone 冻结):")
        model.summary() # 注意看 input/output shape

        callbacks_phase1 = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", 
                patience=15, 
                restore_best_weights=True, 
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", 
                factor=0.5, 
                patience=5,
                min_lr=1e-6, 
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                "baseline_mobilenetv2_phase1.keras",
                monitor="val_accuracy", 
                save_best_only=True, 
                verbose=1
            )
        ]
        
        history_phase1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=50,
            callbacks=callbacks_phase1,
            verbose=1
        )
        
        print("\n=== Phase 2: Fine-tuning (unfreezing last 50 layers) ===")
        
        backbone = model.get_layer("mobilenetv2_backbone")
        backbone.trainable = True
        
        FINE_TUNE_AT = -50
        for layer in backbone.layers[:FINE_TUNE_AT]:
            layer.trainable = False
            
        # 解冻后，Backbone 必须以训练模式运行
        # 我们需要重新构建模型，或者确保 `training` 参数能正确传递
        # Keras 的 .fit() 会自动处理 (training=True)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5), # 微调使用低 LR
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        print("模型摘要 (阶段 2 - 微调):")
        model.summary()

        callbacks_phase2 = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=15, restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                "baseline_mobilenetv2_final.keras",
                monitor="val_accuracy", save_best_only=True, verbose=1
            )
        ]
        
        history_phase2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=30,
            callbacks=callbacks_phase2,
            verbose=1
        )
        
        print("\n=== Saving final model ===")
        model.save(args.model_path)
        
        print("\n=== Evaluating final model on test set ===")
        metrics = benchmark_baseline_model(model, test_ds, batch_size=args.batch_size)
        
        print("\nBaseline Model Performance:")
        print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Test Loss: {metrics['eval_loss']:.4f}")
        # Latency metrics
        if 'single_inference_time' in metrics:
            print(f"  Single Inference Time (s): {metrics['single_inference_time']:.6f}")
        if 'batch_inference_time' in metrics:
            print(f"  Batch Inference Time (s):  {metrics['batch_inference_time']:.6f}")
        # Memory and model complexity
        if 'memory_usage_mb' in metrics:
            print(f"  Inference Memory Delta (MB): {metrics['memory_usage_mb']:.2f}")
        print(f"  Total Parameters: {metrics['parameters']:.0f}")
        print(f"  Model Size (MB): {metrics['model_size_mb']:.2f}")
        if 'flops' in metrics:
            print(f"  Estimated FLOPs: {int(metrics['flops'])}")

        print("\n=== Training Summary ===")
        print(f"Phase 1 - Best validation accuracy: {max(history_phase1.history['val_accuracy']):.4f}")
        val_acc_phase2 = history_phase2.history.get('val_accuracy', [0])
        print(f"Phase 2 - Best validation accuracy: {max(val_acc_phase2): .4f}")
        print(f"Final test accuracy: {metrics['accuracy']:.4f}")
        
        if metrics['accuracy'] >= 0.85:
            print(f"\n✓ SUCCESS: Model achieved {metrics['accuracy']*100:.2f}% accuracy (target: 85%)")
        else:
            print(f"\n✗ Model accuracy {metrics['accuracy']*100:.2f}% is below target (85%)")