"""Diagnostic script to verify dataset preparation and model configuration."""
from __future__ import annotations

import numpy as np
import tf_compat  # noqa: F401
import tensorflow as tf

from baseline_model import prepare_compression_datasets, create_baseline_model


def verify_datasets():
    """Verify that datasets are correctly prepared and normalized."""
    print("=" * 60)
    print("Dataset Verification")
    print("=" * 60)
    
    (x_train, y_train, x_val, y_val, x_test, y_test, calib_data) = prepare_compression_datasets()
    
    print(f"\nTraining set: {x_train.shape}, labels: {y_train.shape}")
    print(f"Validation set: {x_val.shape}, labels: {y_val.shape}")
    print(f"Test set: {x_test.shape}, labels: {y_test.shape}")
    print(f"Calibration set: {calib_data[0].shape}")
    
    # Check normalization
    print(f"\nTraining set stats:")
    print(f"  Mean: {np.mean(x_train):.4f} (should be ~0)")
    print(f"  Std: {np.std(x_train):.4f} (should be ~1)")
    print(f"  Min: {np.min(x_train):.4f}")
    print(f"  Max: {np.max(x_train):.4f}")
    
    print(f"\nValidation set stats:")
    print(f"  Mean: {np.mean(x_val):.4f} (should be ~0)")
    print(f"  Std: {np.std(x_val):.4f} (should be ~1)")
    print(f"  Min: {np.min(x_val):.4f}")
    print(f"  Max: {np.max(x_val):.4f}")
    
    print(f"\nTest set stats:")
    print(f"  Mean: {np.mean(x_test):.4f} (should be ~0)")
    print(f"  Std: {np.std(x_test):.4f} (should be ~1)")
    print(f"  Min: {np.min(x_test):.4f}")
    print(f"  Max: {np.max(x_test):.4f}")
    
    # Check label distribution
    print(f"\nLabel distribution (should be relatively uniform):")
    print(f"  Train unique labels: {len(np.unique(y_train))}")
    print(f"  Val unique labels: {len(np.unique(y_val))}")
    print(f"  Test unique labels: {len(np.unique(y_test))}")
    
    # Verify no data overlap
    train_val_overlap = len(set(map(tuple, x_train.reshape(len(x_train), -1))) & 
                             set(map(tuple, x_val.reshape(len(x_val), -1))))
    print(f"\nData overlap between train and val: {train_val_overlap} (should be 0)")
    
    return True


def test_model_inference():
    """Test model can make predictions on a small batch."""
    print("\n" + "=" * 60)
    print("Model Inference Test")
    print("=" * 60)
    
    model = create_baseline_model()
    
    # Create a small batch
    dummy_batch = np.random.randn(8, 32, 32, 3).astype('float32')
    
    print(f"\nModel summary:")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    print(f"  Total params: {model.count_params():,}")
    
    # Test prediction
    predictions = model.predict(dummy_batch, verbose=0)
    print(f"\nPrediction shape: {predictions.shape}")
    print(f"Prediction sum per sample (should be ~1.0): {predictions.sum(axis=1)[:4]}")
    print(f"Max probability per sample: {predictions.max(axis=1)[:4]}")
    
    return True


def verify_data_pipeline():
    """Verify the tf.data pipeline is correctly configured."""
    print("\n" + "=" * 60)
    print("Data Pipeline Verification")
    print("=" * 60)
    
    from baseline_model import _build_dataset
    
    (x_train, y_train, x_val, y_val, _, _, _) = prepare_compression_datasets()
    
    # Build training dataset WITH augmentation
    train_ds = _build_dataset(x_train, y_train, batch_size=32, augment=True, 
                              shuffle=True, num_classes=100)
    
    # Build validation dataset WITHOUT augmentation
    val_ds = _build_dataset(x_val, y_val, batch_size=32, augment=False, 
                           shuffle=False, num_classes=100)
    
    print("\nSampling from training dataset (with augmentation):")
    for batch_x, batch_y in train_ds.take(1):
        print(f"  Batch shape: {batch_x.shape}, {batch_y.shape}")
        print(f"  X range: [{batch_x.numpy().min():.4f}, {batch_x.numpy().max():.4f}]")
        print(f"  Y shape: {batch_y.shape} (should be [batch, 100] for one-hot)")
        print(f"  Y sum: {batch_y.numpy().sum(axis=1)[:4]} (should be all 1.0)")
    
    print("\nSampling from validation dataset (NO augmentation):")
    for batch_x, batch_y in val_ds.take(1):
        print(f"  Batch shape: {batch_x.shape}, {batch_y.shape}")
        print(f"  X range: [{batch_x.numpy().min():.4f}, {batch_x.numpy().max():.4f}]")
        print(f"  Y shape: {batch_y.shape} (should be [batch, 100] for one-hot)")
        print(f"  Y sum: {batch_y.numpy().sum(axis=1)[:4]} (should be all 1.0)")
    
    return True


if __name__ == "__main__":
    print("\nRunning comprehensive data verification...\n")
    
    verify_datasets()
    test_model_inference()
    verify_data_pipeline()
    
    print("\n" + "=" * 60)
    print("Verification Complete!")
    print("=" * 60)
    print("\nIf all checks passed, you can proceed with training.")
    print("Run: python main.py --train-baseline")
