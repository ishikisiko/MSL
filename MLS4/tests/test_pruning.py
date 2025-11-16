import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from baseline_model import create_baseline_model
from part1_pruning import PruningComparator


def test_magnitude_based_pruning_compiles_and_runs(tmp_path):
    # Build and save a lightweight baseline model for the test
    model = create_baseline_model(num_classes=100)
    model_path = tmp_path / "baseline_model_test.keras"
    model.save(str(model_path))

    # Small synthetic dataset mimicking CIFAR shape and 100 classes
    x_train = np.random.rand(64, 32, 32, 3).astype(np.float32)
    y_train = np.random.randint(0, 100, size=(64,))
    x_val = np.random.rand(16, 32, 32, 3).astype(np.float32)
    y_val = np.random.randint(0, 100, size=(16,))

    pruner = PruningComparator(str(model_path), cache_datasets=False)

    result = pruner.magnitude_based_pruning(
        target_sparsity=0.1,
        fine_tune_epochs=1,
        batch_size=16,
        train_data=(x_train, y_train),
        val_data=(x_val, y_val),
    )

    assert "model" in result
    assert isinstance(result["model"], tf.keras.Model)
    # Cleanup - tmp_path will be removed by pytest fixture