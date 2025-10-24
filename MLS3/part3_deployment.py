"""Deployment helpers for hardware-aware optimisation Track A."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import tensorflow as tf
from tensorflow import keras

from part2_optimizations import representative_dataset_generator


class HardwareOptimizer:
    """Hardware-aware model optimiser for different deployment targets."""

    def __init__(self, target_platform: str):
        self.platform = target_platform
        self.optimization_config = self._get_platform_config()

    def _get_platform_config(self) -> Dict[str, Any]:
        configs = {
            "cpu_x86": {
                "optimization_level": "O3",
                "use_avx": True,
                "thread_count": 4,
                "memory_constraint_mb": 1024,
            },
            "arm_cortex_a": {
                "use_neon": True,
                "fp16_acceleration": True,
                "memory_constraint_mb": 512,
                "power_budget_mw": 2000,
            },
            "arm_cortex_m": {
                "quantization": "int8",
                "memory_constraint_kb": 256,
                "power_budget_mw": 50,
                "use_cmsis_nn": True,
            },
            "gpu_mobile": {
                "use_gpu_delegate": True,
                "fp16_inference": True,
                "memory_constraint_mb": 2048,
                "thermal_throttling": True,
            },
        }
        return configs.get(self.platform, {})

    def optimize_for_platform(self, model: keras.Model, calibration_data: Optional[Any] = None) -> Dict[str, Any]:
        """Apply platform-specific optimisation pipelines and return artefacts."""

        if self.platform == "cpu_x86":
            return self._optimise_for_cpu(model)
        if self.platform == "arm_cortex_a":
            return self._optimise_for_arm_cortex_a(model)
        if self.platform == "arm_cortex_m":
            return self._optimise_for_arm_cortex_m(model, calibration_data)
        if self.platform == "gpu_mobile":
            return self._optimise_for_mobile_gpu(model)

        raise ValueError(f"Unsupported platform: {self.platform}")

    # ------------------------------------------------------------------
    # Platform specialisations
    # ------------------------------------------------------------------
    def _optimise_for_cpu(self, model: keras.Model) -> Dict[str, Any]:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        return {
            "platform": "cpu_x86",
            "artifact_path": self._write_artifact(tflite_model, "model_cpu_x86.tflite"),
            "runtime_configuration": {
                "num_threads": self.optimization_config.get("thread_count", 4),
                "use_xnnpack": True,
            },
            "notes": "XNNPACK-optimised TFLite model for AVX-capable CPUs.",
        }

    def _optimise_for_arm_cortex_a(self, model: keras.Model) -> Dict[str, Any]:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_types = [tf.float16]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        return {
            "platform": "arm_cortex_a",
            "artifact_path": self._write_artifact(tflite_model, "model_cortex_a_fp16.tflite"),
            "runtime_configuration": {
                "delegate": "arm_nn",
                "fp16_enabled": True,
                "neon": self.optimization_config.get("use_neon", True),
            },
            "notes": "FP16-optimised TFLite model suitable for NEON acceleration.",
        }

    def _optimise_for_arm_cortex_m(
        self, model: keras.Model, calibration_data: Optional[Any]
    ) -> Dict[str, Any]:
        if calibration_data is None:
            raise ValueError("calibration_data is required for Cortex-M INT8 conversion")

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.representative_dataset = lambda: representative_dataset_generator(calibration_data)
        tflite_model = converter.convert()

        deployment = deploy_to_tflite_micro(tflite_model)
        return {
            "platform": "arm_cortex_m",
            "artifact_path": deployment["tflite_micro_archive"],
            "runtime_configuration": deployment,
            "notes": "CMSIS-NN compatible INT8 model packaged for TensorFlow Lite Micro.",
        }

    def _optimise_for_mobile_gpu(self, model: keras.Model) -> Dict[str, Any]:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()

        deployment = deploy_to_mobile_gpu(tflite_model)
        return {
            "platform": "gpu_mobile",
            "artifact_path": deployment["tflite_model_path"],
            "runtime_configuration": deployment,
            "notes": "FP16-capable TFLite model configured for GPU delegate execution.",
        }

    @staticmethod
    def _write_artifact(model_bytes: bytes, file_name: str) -> str:
        output_dir = Path("artifacts")
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / file_name
        file_path.write_bytes(model_bytes)
        return str(file_path)


def deploy_to_tflite_micro(
    model_bytes: bytes,
    target_mcu: str = "cortex_m4",
    project_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Package a TFLite model for TensorFlow Lite Micro deployment."""

    project_dir = project_dir or Path("deployments") / f"tflm_{target_mcu}"
    project_dir.mkdir(parents=True, exist_ok=True)

    model_path = project_dir / "model_int8.tflite"
    model_path.write_bytes(model_bytes)

    cmsis_flags = ["-DTF_LITE_USE_CMSIS_NN"] if "cmsis" in target_mcu else []
    build_commands = [
        "cmake -DTARGET_MCU={target} -B build -S .".format(target=target_mcu.upper()),
        "cmake --build build --target flash",
    ]

    return {
        "tflite_micro_archive": str(model_path),
        "project_directory": str(project_dir),
        "build_commands": build_commands,
        "compiler_flags": cmsis_flags + ["-Os", "-ffast-math"],
    }


def deploy_to_mobile_gpu(
    model_bytes: bytes,
    target_gpu: str = "adreno_640",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Prepare deployment bundle for mobile GPU delegates."""

    output_dir = output_dir or Path("deployments") / f"gpu_{target_gpu}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model_gpu_fp16.tflite"
    model_path.write_bytes(model_bytes)

    delegate_config = {
        "use_gpu_delegate": True,
        "precision_loss_allowed": True,
        "target_gpu": target_gpu,
    }

    config_path = output_dir / "gpu_delegate.json"
    config_path.write_text(json.dumps(delegate_config, indent=2))

    adb_commands = [
        f"adb push {model_path} /data/local/tmp/model.tflite",
        "adb shell am start -n com.example.tflitegpu/.MainActivity",
    ]

    return {
        "tflite_model_path": str(model_path),
        "delegate_config": str(config_path),
        "deployment_commands": adb_commands,
    }


def deploy_to_edge_tpu(
    model_bytes: bytes,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Convert and package a model for the Google Coral Edge TPU."""

    output_dir = output_dir or Path("deployments") / "edge_tpu"
    output_dir.mkdir(parents=True, exist_ok=True)

    float_model_path = output_dir / "model_float.tflite"
    float_model_path.write_bytes(model_bytes)

    compiled_model_path = output_dir / "model_edge_tpu.tflite"
    edgetpu_command = f"edgetpu_compiler --out_dir {output_dir} {float_model_path}"

    return {
        "float_model": str(float_model_path),
        "compiled_model": str(compiled_model_path),
        "compilation_command": edgetpu_command,
        "runtime_notes": "Run using libedgetpu with USB or PCIe accelerator.",
    }
