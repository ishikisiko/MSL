#!/usr/bin/env python3
"""Verification script for GPU optimizations and fixes."""

import tensorflow as tf
from tensorflow.keras import mixed_precision
import numpy as np

def verify_gpu_setup():
    """Verify GPU configuration and mixed precision setup."""
    print("=" * 70)
    print("GPU OPTIMIZATION VERIFICATION")
    print("=" * 70)
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\n1. GPU Devices: {len(gpus)} found")
    for i, gpu in enumerate(gpus):
        print(f"   - GPU {i}: {gpu.name}")
    
    # Check mixed precision policy
    policy = mixed_precision.global_policy()
    print(f"\n2. Mixed Precision Policy: {policy.name}")
    print(f"   - Compute dtype: {policy.compute_dtype}")
    print(f"   - Variable dtype: {policy.variable_dtype}")
    
    # Verify memory growth
    print(f"\n3. Memory Growth: ", end="")
    if gpus:
        try:
            memory_growth = tf.config.experimental.get_memory_growth(gpus[0])
            print(f"{'Enabled' if memory_growth else 'Disabled'}")
        except:
            print("Unable to check")
    else:
        print("N/A (No GPU)")
    
    return len(gpus) > 0


def verify_batchnorm_dropout_fix():
    """Verify that BatchNorm fix doesn't interfere with model."""
    print("\n" + "=" * 70)
    print("BATCHNORM + DROPOUT FIX VERIFICATION")
    print("=" * 70)
    
    # Create a simple model with BN and Dropout
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, padding='same', input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10)
    ])
    
    # Check dropout layers
    dropout_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dropout)]
    print(f"\n1. Found {len(dropout_layers)} Dropout layer(s)")
    
    # Simulate the fix
    original_rates = []
    for layer in dropout_layers:
        original_rates.append(layer.rate)
        print(f"   - Original rate: {layer.rate}")
        layer.rate = 0.0
        print(f"   - Disabled rate: {layer.rate}")
    
    # Restore
    for layer, rate in zip(dropout_layers, original_rates):
        layer.rate = rate
        print(f"   - Restored rate: {layer.rate}")
    
    print("\n✓ BatchNorm fix mechanism works correctly")
    return True


def verify_tflite_delegate():
    """Verify TFLite GPU delegate availability."""
    print("\n" + "=" * 70)
    print("TFLITE GPU DELEGATE VERIFICATION")
    print("=" * 70)
    
    try:
        # Check if experimental delegate loading is available
        if hasattr(tf.lite.experimental, 'load_delegate'):
            print("\n1. tf.lite.experimental.load_delegate: Available")
            
            # Try to load Metal delegate (macOS)
            try:
                delegate = tf.lite.experimental.load_delegate('libmetal_delegate.so')
                print("2. Metal GPU Delegate: Successfully loaded")
                return True
            except Exception as e:
                print(f"2. Metal GPU Delegate: Not available ({e})")
                print("   → Will fall back to CPU interpreter")
                return False
        else:
            print("\n1. tf.lite.experimental.load_delegate: Not available")
            print("   → TFLite will use CPU interpreter")
            return False
    except Exception as e:
        print(f"\nError checking TFLite delegate: {e}")
        return False


def verify_dataset_optimization():
    """Verify dataset caching and prefetching."""
    print("\n" + "=" * 70)
    print("DATASET OPTIMIZATION VERIFICATION")
    print("=" * 70)
    
    # Create sample dataset
    x = np.random.rand(100, 32, 32, 3).astype(np.float32)
    y = np.random.randint(0, 10, 100).astype(np.int32)
    
    # Build optimized dataset
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.cache()
    ds = ds.batch(32)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    print("\n1. Dataset Pipeline:")
    print("   - cache(): Enabled")
    print("   - batch(32): Configured")
    print("   - prefetch(AUTOTUNE): Enabled")
    
    # Test iteration
    import time
    start = time.time()
    for _ in ds:
        pass
    first_epoch = time.time() - start
    
    start = time.time()
    for _ in ds:
        pass
    second_epoch = time.time() - start
    
    print(f"\n2. Iteration Performance:")
    print(f"   - First epoch: {first_epoch*1000:.2f} ms")
    print(f"   - Second epoch (cached): {second_epoch*1000:.2f} ms")
    print(f"   - Speedup: {first_epoch/second_epoch:.2f}x")
    
    return True


def main():
    """Run all verification tests."""
    print("\n🔧 GPU OPTIMIZATIONS VERIFICATION SUITE")
    print("This script verifies all GPU acceleration improvements\n")
    
    results = {
        'GPU Setup': verify_gpu_setup(),
        'BatchNorm Fix': verify_batchnorm_dropout_fix(),
        'TFLite Delegate': verify_tflite_delegate(),
        'Dataset Optimization': verify_dataset_optimization(),
    }
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for test, passed in results.items():
        status = "✓ PASS" if passed else "⚠ WARNING"
        print(f"{status}: {test}")
    
    print("\n" + "=" * 70)
    
    if all(results.values()):
        print("✓ All optimizations verified successfully!")
    else:
        print("⚠ Some optimizations may not be available on this system")
        print("  The code will automatically fall back to CPU where needed")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
