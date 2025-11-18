"""Runtime helpers to keep TensorFlow compatible with TF-MOT pruning utilities.

This module must be imported before any ``import tensorflow as tf`` statements so
that the environment flag is applied early in the process.
"""
from __future__ import annotations

import os
from typing import Final

# =====================================================================
# CRITICAL: Disable XLA FIRST before any other operations
# Must be the VERY FIRST thing that happens in this module
# =====================================================================
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0 --tf_xla_cpu_global_jit=false --tf_xla_clustering_debug=false'
os.environ['TF_DISABLE_XLA_JIT'] = '1'
os.environ['TF_XLA_AUTO_JIT'] = '0'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir='  # Disable XLA GPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Disable all graph optimizations that might trigger XLA
os.environ['TF_DISABLE_MKL'] = '1'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'

_FLAG_NAME: Final[str] = "TF_USE_LEGACY_KERAS"

# TensorFlow Model Optimization currently depends on legacy tf.keras behavior.
# Setting the env var ahead of TensorFlow imports keeps Functional models
# compatible with prune_low_magnitude when running on newer Keras builds.
if os.environ.get(_FLAG_NAME) not in {"1", "true", "True"}:
    os.environ.setdefault(_FLAG_NAME, "1")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging noise

# Fix EfficientNet GPU layout optimization issues
# CRITICAL: Completely disable XLA JIT compilation
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0'
os.environ['TF_DISABLE_XLA_JIT'] = '1'
os.environ['TF_XLA_AUTO_JIT'] = '0'

# Disable layout and graph optimizations that cause issues with EfficientNet
os.environ['TF_ENABLE_LAYOUT_OPT'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '0'
os.environ['TF_CUDNN_DETERMINISTIC'] = '0'

# Disable function inlining and autograph to reduce retracing
os.environ['TF_ENABLE_AUTOGRAPH'] = '0'
os.environ['TF_ENABLE_FUNCTION_INLINING'] = '0'

print("=" * 70)
print("XLA JIT COMPILATION DISABLED - EfficientNet Compatibility Mode")
print("=" * 70)
