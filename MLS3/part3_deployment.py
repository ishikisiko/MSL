"""Deployment helpers for Track A: convert and export models to target runtimes.

Includes TFLite conversion, basic TFLite Micro export, and stubs for Edge-TPU/GPU
deployment steps.
"""

from __future__ import annotations

from typing import Dict, Optional

import os
import shutil
import subprocess
import tensorflow as tf
from tensorflow import keras


def convert_to_tflite(model: keras.Model, save_path: str = "model.tflite", optimizations=None) -> str:
    """Convert a Keras model to TFLite and save to disk.

    optimizations: list of tf.lite.Optimize enums or None
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if optimizations:
        converter.optimizations = optimizations
    tflite_model = converter.convert()
    with open(save_path, "wb") as f:
        f.write(tflite_model)
    return save_path


def deploy_to_tflite_micro(tflite_path: str, output_c_path: str = "model_data.cc") -> str:
    """Create a C source file that embeds the TFLite flatbuffer for TFLite-Micro.

    This is a very small helper that writes a C array. It's not optimized but
    suitable for inclusion in microcontroller firmware projects.
    """
    with open(tflite_path, "rb") as f:
        data = f.read()

    # Create C array
    array_lines = []
    for i in range(0, len(data), 12):
        chunk = data[i : i + 12]
        array_lines.append(
            ", ".join(str(b) for b in chunk)
        )

    with open(output_c_path, "w", encoding="utf-8") as f:
        f.write("#include <cstdint>\n\n")
        f.write(f"extern const unsigned char g_model[] = {{\n")
        for line in array_lines:
            f.write("  " + line + ",\n")
        f.write("};\n")
        f.write(f"extern const unsigned int g_model_len = {len(data)};\n")

    return output_c_path


def deploy_to_edge_tpu(tflite_path: str, output_path: Optional[str] = None) -> Dict[str, str]:
    """Attempt to compile a TFLite model for Edge TPU using edgetpu_compiler.

    This function calls the `edgetpu_compiler` binary if available on PATH.
    """
    if output_path is None:
        output_path = os.path.splitext(tflite_path)[0] + "_edgetpu.tflite"

    compiler = shutil.which("edgetpu_compiler")
    if compiler is None:
        return {"status": "missing_compiler", "message": "edgetpu_compiler not found on PATH"}

    try:
        subprocess.check_call([compiler, tflite_path, "-o", os.path.dirname(output_path)])
        return {"status": "ok", "compiled_model": output_path}
    except subprocess.CalledProcessError as exc:
        return {"status": "error", "message": str(exc)}


def deploy_to_mobile_gpu(tflite_path: str, runtime: str = "gpu_delegate") -> Dict[str, str]:
    """Stub to indicate how to prepare model for mobile GPU delegates.

    In practice, use the platform's delegate (TFLite GPU delegate / NNAPI / CoreML) and
    profile on-device. This helper returns metadata and a recommended path.
    """
    return {"status": "ok", "tflite": tflite_path, "note": "Use platform-specific delegate when executing on-device."}


__all__ = ["convert_to_tflite", "deploy_to_tflite_micro", "deploy_to_edge_tpu", "deploy_to_mobile_gpu"]
