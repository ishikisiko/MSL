"""
Quick test script to verify quantization fixes
"""
import tf_compat  # Import this FIRST to configure tf_keras compatibility
import tensorflow as tf
import numpy as np
from baseline_model import load_baseline_model
from part2_quantization import QuantizationPipeline

print("="*60)
print("Testing Quantization Fixes")
print("="*60)

# Load baseline model
print("\n1. Loading baseline model...")
model = load_baseline_model("models/baseline_model.keras")
print(f"   Model loaded: {model.name}")

# Create quantization pipeline
print("\n2. Creating quantization pipeline...")
pipeline = QuantizationPipeline(model, cache_datasets=True, default_batch_size=32)
print("   Pipeline created")

# Test PTQ with debug output
print("\n3. Testing Post-Training Quantization...")
print("   This will show detailed debug information")
print("-"*60)

results = pipeline.standard_tflite_quantization(
    quantization_type="post_training",
    target_platform="generic"
)

print("\n" + "="*60)
print("Test Results Summary")
print("="*60)

if "post_training_quantization" in results:
    ptq = results["post_training_quantization"]
    print(f"✓ PTQ completed successfully")
    print(f"  Accuracy: {ptq.get('accuracy', 'N/A'):.4f}")
    print(f"  Model size: {ptq.get('model_size_mb', 'N/A'):.2f} MB")
    print(f"  Model path: {ptq.get('model_path', 'N/A')}")
    
    # Check if accuracy is reasonable (should be > 0.2 for CIFAR-100)
    acc = ptq.get('accuracy', 0)
    if acc > 0.2:
        print("\n✓ PASS: Accuracy is reasonable (> 20%)")
    else:
        print("\n✗ FAIL: Accuracy too low - likely a quantization issue")
        print("  Expected: > 0.2 (20% for CIFAR-100)")
        print(f"  Got: {acc:.4f}")
else:
    print("✗ PTQ failed")
    if "error" in results:
        print(f"  Error: {results['error']}")

print("\n" + "="*60)
