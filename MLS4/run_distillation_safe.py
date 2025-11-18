#!/usr/bin/env python
"""Safe distillation runner that ensures XLA is completely disabled."""

import os
import sys

# CRITICAL: Set these BEFORE any other imports
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0'
os.environ['TF_DISABLE_XLA_JIT'] = '1'
os.environ['TF_XLA_AUTO_JIT'] = '0'
os.environ['TF_ENABLE_LAYOUT_OPT'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '0'
os.environ['TF_CUDNN_DETERMINISTIC'] = '0'
os.environ['TF_ENABLE_AUTOGRAPH'] = '0'
os.environ['TF_ENABLE_FUNCTION_INLINING'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

print("=" * 70)
print("SAFE DISTILLATION MODE - XLA JIT COMPLETELY DISABLED")
print("=" * 70)
print(f"TF_XLA_FLAGS = {os.environ['TF_XLA_FLAGS']}")
print(f"TF_DISABLE_XLA_JIT = {os.environ['TF_DISABLE_XLA_JIT']}")
print("=" * 70)
print()

# Now it's safe to import TensorFlow and run the script
import tensorflow as tf

# Double-check: Force disable JIT at TF level
try:
    tf.config.optimizer.set_jit(False)
    print("✓ TensorFlow JIT compilation disabled at runtime")
except Exception as e:
    print(f"⚠ Warning: Could not disable JIT at runtime: {e}")

# Check XLA status
print(f"✓ TensorFlow version: {tf.__version__}")
print(f"✓ Keras version: {tf.keras.__version__}")
print()

# Modify sys.argv to add distillation-only flags
sys.argv = [
    'main.py',
    '--onlyd',
    '--distillation-batch-size', '32',
    '--seed', '42',
    '--skip-baseline',  # Don't retrain baseline
]

# Import and run main
from main import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDistillation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nDistillation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
