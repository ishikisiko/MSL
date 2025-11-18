"""Runtime helpers to keep TensorFlow compatible with TF-MOT pruning utilities.

This module must be imported before any ``import tensorflow as tf`` statements so
that the environment flag is applied early in the process.
"""
from __future__ import annotations

import os
from typing import Final

_FLAG_NAME: Final[str] = "TF_USE_LEGACY_KERAS"

# TensorFlow Model Optimization currently depends on legacy tf.keras behavior.
# Setting the env var ahead of TensorFlow imports keeps Functional models
# compatible with prune_low_magnitude when running on newer Keras builds.
if os.environ.get(_FLAG_NAME) not in {"1", "true", "True"}:
    os.environ.setdefault(_FLAG_NAME, "1")

# Fix EfficientNet GPU layout optimization issues
# Disable layout optimization that causes issues with EfficientNet dropout layers
os.environ['TF_ENABLE_LAYOUT_OPT'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '0'
os.environ['TF_CUDNN_DETERMINISTIC'] = '0'
# Disable XLA JIT compilation to avoid layout errors
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
