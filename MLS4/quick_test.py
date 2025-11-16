"""Quick test to ensure model can train for a few epochs without errors."""
from __future__ import annotations

import tf_compat  # noqa: F401
import tensorflow as tf
from baseline_model import train_baseline_model

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Running quick training test (3 epochs)...")
    print("="*70 + "\n")
    
    # Train for just 3 epochs to verify everything works
    result = train_baseline_model(
        epochs=3,
        batch_size=256,
        output_path="test_model.keras",
        checkpoint_path="checkpoints/test_checkpoint.keras",
        seed=42
    )
    
    print("\n" + "="*70)
    print("Test completed successfully!")
    print("="*70)
    print(f"\nFinal test accuracy: {result['test_accuracy']:.4f}")
    print(f"\nHistory keys: {list(result['history'].keys())}")
    
    if 'val_accuracy' in result['history']:
        final_val_acc = result['history']['val_accuracy'][-1]
        print(f"Final validation accuracy: {final_val_acc:.4f}")
        
        if final_val_acc > 0.15:
            print("✅ Validation accuracy looks healthy (>15%)")
        else:
            print("⚠️  Validation accuracy still low, may need more investigation")
    
    print("\nYou can now run full training with:")
    print("  python main.py --train-baseline")
