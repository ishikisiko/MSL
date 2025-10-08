# part1_tensorflow.py
# Train model (variable batch) -> create inference model (batch=1) -> save inference model
# Uses a serializable custom layer FixedBatchReshape to avoid Lambda deserialization issues.

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from memory_profiler import profile

MODEL_OUT = "mnist_cnn_model.keras"
TRAINED_WEIGHTS = "mnist_cnn_weights.weights.h5"  # 保证以 .weights.h5 结尾避免 Keras 的历史检查

# Serializable custom layer that reshapes to a fixed batch=1 target.
class FixedBatchReshape(keras.layers.Layer):
    def __init__(self, target_dim, **kwargs):
        super().__init__(**kwargs)
        self.target_dim = int(target_dim)

    def call(self, inputs):
        # Force explicit batch dimension = 1 in the reshape target
        return tf.reshape(inputs, [1, self.target_dim])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"target_dim": self.target_dim})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def create_train_model():
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(28, 28, 1)),
        keras.layers.Conv2D(8, (3, 3), activation='relu'),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        # training-time reshape; batch remaining dynamic here
        keras.layers.Reshape((13 * 13 * 8,)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(10)  # logits
    ])
    return model

def create_inference_model():
    # Fixed batch=1 model. Use FixedBatchReshape to produce explicit [1, 1352] shape.
    model = keras.Sequential([
        keras.layers.InputLayer(batch_input_shape=(1, 28, 28, 1)),
        keras.layers.Conv2D(8, (3, 3), activation='relu'),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        FixedBatchReshape(13 * 13 * 8),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(10)
    ])
    return model

def prepare_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test  = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)

def train_and_save(quick=False):
    (x_train, y_train), (x_test, y_test) = prepare_data()
    train_model = create_train_model()
    train_model.summary()
    train_model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    epochs = 1 if quick else 5
    train_model.fit(x_train, y_train, epochs=epochs, batch_size=128, validation_split=0.1)

    # 保存训练权重
    train_model.save_weights(TRAINED_WEIGHTS)

    # 构建推理模型并拷贝权重
    inf_model = create_inference_model()
    inf_model.set_weights(train_model.get_weights())
    # 保存为 .keras（单文件）格式；该文件在加载时需要 FixedBatchReshape 在作用域中
    inf_model.save(MODEL_OUT, include_optimizer=False)
    print(f"Saved inference model to: {MODEL_OUT}")
    print(f"Saved intermediate weights to: {TRAINED_WEIGHTS}")

@profile
def run_inference():
    print("\nRunning inference with memory profiling...")
    # 加载自定义层作用域中的模型
    custom_objects = {"FixedBatchReshape": FixedBatchReshape}
    model = keras.models.load_model(MODEL_OUT, custom_objects=custom_objects)
    
    (_, _), (x_test, _) = prepare_data()
    
    # 取前5个样本进行推理
    for i in range(5):
        sample = x_test[i:i+1] # Shape (1, 28, 28, 1)
        prediction = model.predict(sample)
        print(f"Inference on sample {i}, prediction: {np.argmax(prediction)}")

if __name__ == "__main__":
    # 训练并生成 inference 模型（如果已存在会跳过）
    if not os.path.exists(MODEL_OUT):
        print("No inference model found — training and creating it now.")
        train_and_save(quick=False)
    else:
        print(f"Found existing inference model '{MODEL_OUT}'. To retrain, delete it and re-run.")

    run_inference()
