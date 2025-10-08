import numpy as np
import tensorflow as tf
from tensorflow import keras


def configure_gpu_memory_growth():
    """Enable memory growth for all detected GPUs to avoid OOM on allocation."""

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[GPU] 未检测到可用的 GPU，训练将回退到 CPU。")
        return False

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as exc:  # Memory growth must be set before GPUs are initialized.
        print(f"[GPU] 设置显存按需分配失败：{exc}.")
        return False

    print(f"[GPU] 已为 {len(gpus)} 块 GPU 启用显存按需分配。")
    return True


def create_distribution_strategy():
    """Create the best-fit distribution strategy for the current hardware."""

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            strategy = tf.distribute.MirroredStrategy()
            print(
                f"[GPU] 使用 MirroredStrategy，副本数：{strategy.num_replicas_in_sync}."
            )
            return strategy
        except RuntimeError as exc:
            print(f"[GPU] 创建 MirroredStrategy 失败：{exc}，改用默认策略。")

    print("[GPU] 使用默认策略（通常为 CPU 单进程）。")
    return tf.distribute.get_strategy()


def create_baseline_model(input_shape=(32, 32, 3), num_classes=10, strategy=None):
    """
    Create a moderately complex CNN for CIFAR-10 classification.

    Args:
        input_shape (tuple): Shape of the input images.
        num_classes (int): Number of target classes.
        strategy (tf.distribute.Strategy | None): Optional distribution strategy.

    Returns:
        tf.keras.Model: Compiled model ready for training.
    """
    def build_model():
        model = keras.Sequential(
            [
                keras.layers.Input(shape=input_shape),
                # Block 1
                keras.layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal"),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                keras.layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal"),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                # Block 2
                keras.layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal"),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                keras.layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal"),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                # Block 3
                keras.layers.Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal"),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                keras.layers.Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal"),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                # Classifier
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal"),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        return model

    if strategy is None:
        return build_model()

    with strategy.scope():
        return build_model()


def load_and_preprocess_data(batch_size=128, validation_split=0.1, seed=42):
    """
    Load and preprocess CIFAR-10 dataset with normalization and augmentation.

    Args:
        batch_size (int): Batch size for the datasets.
        validation_split (float): Fraction of the training data to reserve for validation.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train = y_train.squeeze().astype(np.int64)
    y_test = y_test.squeeze().astype(np.int64)

    num_train = x_train.shape[0]
    val_size = int(num_train * validation_split)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_train)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    x_val = x_train[val_indices]
    y_val = y_train[val_indices]
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]

    data_augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )

    autotune = tf.data.AUTOTUNE

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=len(x_train), seed=seed)
        .batch(batch_size)
        .map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=autotune)
        .prefetch(autotune)
    )

    val_dataset = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(batch_size)
        .prefetch(autotune)
    )

    test_dataset = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .prefetch(autotune)
    )

    return train_dataset, val_dataset, test_dataset


def train_baseline_model(model, train_dataset, val_dataset, test_dataset, max_epochs=50):
    """
    Train the baseline model with callbacks for regularization and monitoring.

    Args:
        model (tf.keras.Model): Compiled model ready for training.
        train_dataset (tf.data.Dataset): Training dataset with augmentation.
        val_dataset (tf.data.Dataset): Validation dataset for monitoring.
        test_dataset (tf.data.Dataset): Test dataset for final evaluation.
        max_epochs (int): Maximum number of training epochs.

    Returns:
        tuple: (model, training_history, training_metrics)
    """

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
        ),
        keras.callbacks.ModelCheckpoint(
            filepath="baseline_checkpoint.keras",
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=max_epochs,
        callbacks=callbacks,
    )

    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
    metrics = {"test_loss": test_loss, "test_accuracy": test_accuracy}

    return model, history, metrics


if __name__ == "__main__":
    configure_gpu_memory_growth()
    strategy = create_distribution_strategy()

    train_ds, val_ds, test_ds = load_and_preprocess_data()
    model = create_baseline_model(strategy=strategy)
    model, history, metrics = train_baseline_model(model, train_ds, val_ds, test_ds)
    model.save("baseline_model.keras")
    print(f"Baseline model parameters: {model.count_params():,}")
    print(f"Baseline test accuracy: {metrics['test_accuracy']:.4f}")