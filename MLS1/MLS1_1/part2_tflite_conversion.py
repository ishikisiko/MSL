# part2_tflite_conversion.py
# Load inference model saved by part1, convert to TFLite (plain/quantized), analyze sizes, and test accuracy.

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys
import math
import tempfile
from memory_profiler import profile

# Must match the FixedBatchReshape class in part1 (so model can be deserialized)
class FixedBatchReshape(keras.layers.Layer):
    def __init__(self, target_dim, **kwargs):
        super().__init__(**kwargs)
        self.target_dim = int(target_dim)

    def call(self, inputs):
        return tf.reshape(inputs, [1, self.target_dim])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"target_dim": self.target_dim})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)

MODEL_IN = "mnist_cnn_model.keras"
TFLITE_OUT = "mnist_model.tflite"
TFLITE_QUANT_OUT = "mnist_model_quantized.tflite"

def convert_to_tflite(model_path, quantize=False):
    # Load the .keras model with custom layer class available in scope
    model = tf.keras.models.load_model(model_path, compile=False, custom_objects={"FixedBatchReshape": FixedBatchReshape})
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if not quantize:
        # default conversion (no quantization)
        tflite_model = converter.convert()
        return tflite_model

    # For full integer quantization we need a representative dataset.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)

    def representative_gen():
        for i in range(100):  # 100 samples is usually enough
            img = x_train[i]
            img = np.expand_dims(img, 0).astype(np.float32)
            yield [img]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_gen
    # Ensure full integer if possible:
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # set input/output types to int8
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    return tflite_model

def analyze_model_size(tf_model_path, tflite_model_data):
    # TF model may be a .keras file or SavedModel dir; compute appropriate size
    if os.path.isdir(tf_model_path):
        total = 0
        for root, _, files in os.walk(tf_model_path):
            for f in files:
                fp = os.path.join(root, f)
                total += os.path.getsize(fp)
        tf_size = total
    else:
        tf_size = os.path.getsize(tf_model_path)
    tflite_size = len(tflite_model_data)
    ratio = tflite_size / tf_size if tf_size > 0 else float("nan")
    print(f"TF model size: {tf_size} bytes")
    print(f"TFLite model size: {tflite_size} bytes")
    print(f"Compression ratio (tflite/tf): {ratio:.3f}")

@profile
def test_tflite_accuracy(tflite_model_data, x_test, y_test, max_eval=2000):
    # Create interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model_data)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    quant_params = input_details.get("quantization", (0.0, 0))
    scale, zero_point = quant_params

    total = 0
    correct = 0
    n = min(len(x_test), max_eval)
    for i in range(n):
        inp = x_test[i:i+1].astype(np.float32)
        if scale != 0.0:
            # quantize input to int8/uint8
            inp_q = inp / scale + zero_point
            inp_q = np.round(inp_q).astype(input_details["dtype"])
            interpreter.set_tensor(input_details["index"], inp_q)
        else:
            interpreter.set_tensor(input_details["index"], inp)

        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])
        # if output is quantized, dequantize
        out_quant = output_details.get("quantization", (0.0, 0))
        out_scale, out_zp = out_quant
        if out_scale != 0.0:
            out = out_scale * (out.astype(np.float32) - out_zp)
        pred = np.argmax(out, axis=1)[0]
        if pred == int(y_test[i]):
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    print(f"Evaluated {total} examples — TFLite accuracy: {acc:.4f}")
    return acc

if __name__ == "__main__":
    if not os.path.exists(MODEL_IN):
        print(f"Model '{MODEL_IN}' not found. Run part1_tensorflow.py first to create it.")
        sys.exit(1)

    print("Converting plain (no-quant) TFLite...")
    tflite_plain = convert_to_tflite(MODEL_IN, quantize=False)
    with open(TFLITE_OUT, "wb") as f:
        f.write(tflite_plain)
    print(f"Wrote {TFLITE_OUT}")

    print("Converting quantized (full int8) TFLite...")
    try:
        tflite_quant = convert_to_tflite(MODEL_IN, quantize=True)
        with open(TFLITE_QUANT_OUT, "wb") as f:
            f.write(tflite_quant)
        print(f"Wrote {TFLITE_QUANT_OUT}")
    except Exception as e:
        print("Quantized conversion failed:", e)
        tflite_quant = None

    # Analyze sizes
    analyze_model_size(MODEL_IN, tflite_plain)
    if tflite_quant is not None:
        analyze_model_size(MODEL_IN, tflite_quant)

    # Quick accuracy test using test data (subset)
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1)

    print("\nTesting plain TFLite accuracy (subset)...")
    test_tflite_accuracy(tflite_plain, x_test, y_test, max_eval=2000)
    if tflite_quant is not None:
        print("\nTesting quantized TFLite accuracy (subset)...")
        test_tflite_accuracy(tflite_quant, x_test, y_test, max_eval=2000)
