"""Example script demonstrating the full optimization pipeline.

This script loads the baseline model, creates optimized variants, applies
quantization, and generates performance comparisons.

Usage:
    python run_optimizations.py
"""

from __future__ import annotations

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Import our modules
from part1_baseline_model import load_and_preprocess_data
from part2_optimizations import (
    create_latency_optimized_model,
    create_memory_optimized_model,
    create_energy_optimized_model,
    representative_dataset_generator,
    post_training_quantization,
    dynamic_range_quantization,
)
from performance_profiler import (
    profile_model_comprehensive,
    compare_models,
    print_profiling_results,
)
from part3_modeling import PlatformPerformanceModel, simulate_arm_performance
from part3_deployment import convert_to_tflite, deploy_to_tflite_micro


def ensure_directories():
    """Create necessary output directories."""
    os.makedirs("optimized_models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)


def load_baseline_model():
    """Load the trained baseline model."""
    baseline_path = "baseline_mobilenetv2.keras"
    if not os.path.exists(baseline_path):
        print(f"\n⚠ Baseline model not found at {baseline_path}")
        print("Please run part1_baseline_model.py first to train the baseline model.")
        sys.exit(1)
    
    print(f"\n✓ Loading baseline model from {baseline_path}")
    model = keras.models.load_model(baseline_path)
    return model


def create_and_train_optimized_variants(train_ds, val_ds):
    """Create and fine-tune optimized model variants.
    
    Note: For simplicity, this example doesn't fully train optimized models.
    In practice, you should fine-tune each variant on your dataset.
    """
    print("\n" + "="*70)
    print("Creating Optimized Model Variants")
    print("="*70)
    
    variants = {}
    
    # 1. Latency-optimized model
    print("\n1. Creating latency-optimized model (alpha=0.5, 128x128)...")
    latency_model = create_latency_optimized_model(
        input_shape=(128, 128, 3),
        num_classes=10,
        alpha=0.5
    )
    latency_model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Quick fine-tuning (you may want to train longer in practice)
    print("   Fine-tuning latency model (3 epochs)...")
    latency_model.fit(
        train_ds.map(lambda x, y: (tf.image.resize(x, [128, 128]), y)),
        validation_data=val_ds.map(lambda x, y: (tf.image.resize(x, [128, 128]), y)),
        epochs=3,
        verbose=1
    )
    latency_model.save("optimized_models/latency_optimized.keras")
    variants["latency"] = latency_model
    
    # 2. Memory-optimized model
    print("\n2. Creating memory-optimized model (alpha=0.35, 96x96)...")
    memory_model = create_memory_optimized_model(
        input_shape=(96, 96, 3),
        num_classes=10,
        width_multiplier=0.35
    )
    memory_model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("   Fine-tuning memory model (3 epochs)...")
    memory_model.fit(
        train_ds.map(lambda x, y: (tf.image.resize(x, [96, 96]), y)),
        validation_data=val_ds.map(lambda x, y: (tf.image.resize(x, [96, 96]), y)),
        epochs=3,
        verbose=1
    )
    memory_model.save("optimized_models/memory_optimized.keras")
    variants["memory"] = memory_model
    
    # 3. Energy-optimized model
    print("\n3. Creating energy-optimized model (alpha=0.75, 96x96)...")
    energy_model = create_energy_optimized_model(
        input_shape=(96, 96, 3),
        num_classes=10
    )
    energy_model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("   Fine-tuning energy model (3 epochs)...")
    energy_model.fit(
        train_ds.map(lambda x, y: (tf.image.resize(x, [96, 96]), y)),
        validation_data=val_ds.map(lambda x, y: (tf.image.resize(x, [96, 96]), y)),
        epochs=3,
        verbose=1
    )
    energy_model.save("optimized_models/energy_optimized.keras")
    variants["energy"] = energy_model
    
    print("\n✓ All optimized variants created and saved.")
    return variants


def apply_quantization(model, model_name, train_ds):
    """Apply quantization techniques to a model."""
    print(f"\n" + "="*70)
    print(f"Applying Quantization to {model_name}")
    print("="*70)
    
    # Prepare representative dataset
    x_samples = []
    for batch in train_ds.take(20):
        x_samples.extend(batch[0].numpy())
    
    rep_gen = representative_dataset_generator(x_samples, num_samples=100)
    
    # 1. Dynamic range quantization
    print(f"\n1. Applying dynamic range quantization...")
    dynamic_path = f"optimized_models/{model_name}_dynamic.tflite"
    dynamic_range_quantization(model, save_path=dynamic_path)
    print(f"   ✓ Saved to {dynamic_path}")
    
    # 2. Post-training quantization (INT8)
    print(f"\n2. Applying post-training quantization (INT8)...")
    try:
        # Recreate generator for each use
        rep_gen = representative_dataset_generator(x_samples, num_samples=100)
        ptq_path = f"optimized_models/{model_name}_ptq_int8.tflite"
        post_training_quantization(model, rep_gen, save_path=ptq_path)
        print(f"   ✓ Saved to {ptq_path}")
    except Exception as e:
        print(f"   ⚠ PTQ failed: {e}")
    
    return dynamic_path


def benchmark_all_models(baseline_model, variants, test_ds):
    """Benchmark all models and generate comparison reports."""
    print("\n" + "="*70)
    print("Benchmarking All Models")
    print("="*70)
    
    platform_config = {
        "power_budget_w": 5.0,
        "memory_budget_mb": 1024,
        "tdp_watts": 10.0,
    }
    
    results = {}
    
    # Benchmark baseline
    print("\n[1/4] Profiling baseline model...")
    baseline_results = profile_model_comprehensive(
        baseline_model, test_ds, platform_config
    )
    results["baseline"] = baseline_results
    print_profiling_results(baseline_results, "Baseline Model")
    
    # Benchmark latency-optimized
    print("\n[2/4] Profiling latency-optimized model...")
    latency_test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, [128, 128]), y))
    latency_results = profile_model_comprehensive(
        variants["latency"], latency_test_ds, platform_config
    )
    results["latency"] = latency_results
    print_profiling_results(latency_results, "Latency-Optimized Model")
    
    # Benchmark memory-optimized
    print("\n[3/4] Profiling memory-optimized model...")
    memory_test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, [96, 96]), y))
    memory_results = profile_model_comprehensive(
        variants["memory"], memory_test_ds, platform_config
    )
    results["memory"] = memory_results
    print_profiling_results(memory_results, "Memory-Optimized Model")
    
    # Benchmark energy-optimized
    print("\n[4/4] Profiling energy-optimized model...")
    energy_test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, [96, 96]), y))
    energy_results = profile_model_comprehensive(
        variants["energy"], energy_test_ds, platform_config
    )
    results["energy"] = energy_results
    print_profiling_results(energy_results, "Energy-Optimized Model")
    
    return results


def generate_comparison_report(baseline_results, all_results):
    """Generate and print comparison report."""
    print("\n" + "="*70)
    print("Performance Comparison Summary")
    print("="*70)
    
    print(f"\n{'Metric':<30} {'Baseline':<15} {'Latency':<15} {'Memory':<15} {'Energy':<15}")
    print("-" * 95)
    
    metrics_to_compare = [
        ("Parameters", "parameters", "{:.0f}"),
        ("Model Size (MB)", "model_size_mb", "{:.2f}"),
        ("Mean Latency (ms)", "mean_latency_ms", "{:.2f}"),
        ("Throughput (FPS)", "throughput_fps", "{:.2f}"),
        ("Energy (mJ)", "total_energy_mj", "{:.2f}"),
        ("Accuracy", "accuracy", "{:.4f}"),
    ]
    
    for label, key, fmt in metrics_to_compare:
        baseline_val = baseline_results.get(key, 0)
        latency_val = all_results["latency"].get(key, 0)
        memory_val = all_results["memory"].get(key, 0)
        energy_val = all_results["energy"].get(key, 0)
        
        print(f"{label:<30} {fmt.format(baseline_val):<15} {fmt.format(latency_val):<15} "
              f"{fmt.format(memory_val):<15} {fmt.format(energy_val):<15}")
    
    print("\n" + "="*70)
    
    # Generate individual comparisons
    for name in ["latency", "memory", "energy"]:
        comparison = compare_models(baseline_results, all_results[name], name)
        print(f"\n{name.upper()} Model vs Baseline:")
        if "speedup" in comparison:
            print(f"  Speedup: {comparison['speedup']:.2f}x")
        for metric, improvement in comparison.get("improvements", {}).items():
            print(f"  {metric}: {improvement:+.2f}% improvement")
        if "accuracy_drop" in comparison.get("degradations", {}):
            print(f"  Accuracy drop: {comparison['degradations']['accuracy_drop']:.4f}")


def demonstrate_track_b_simulation(baseline_model):
    """Demonstrate Track B performance modeling and simulation."""
    print("\n" + "="*70)
    print("Track B: Performance Modeling & Simulation")
    print("="*70)
    
    # Define platform specifications
    platforms = {
        "ARM Cortex-A78": {
            "frequency_ghz": 2.4,
            "peak_gflops": 200,
            "memory_bandwidth_gbps": 15,
            "tdp_watts": 5.0,
            "power_budget_w": 5.0,
            "compute_efficiency_ops_per_joule": 5e9,
            "simulation_overhead_scale": 2.0,
        },
        "ARM Cortex-M7": {
            "frequency_ghz": 0.4,
            "peak_gflops": 2.0,
            "memory_bandwidth_gbps": 0.5,
            "tdp_watts": 0.1,
            "power_budget_w": 0.1,
            "compute_efficiency_ops_per_joule": 1e8,
            "simulation_overhead_scale": 5.0,
        },
    }
    
    print("\nSimulating performance on different platforms:\n")
    
    for platform_name, specs in platforms.items():
        print(f"{platform_name}:")
        perf = simulate_arm_performance(baseline_model, specs)
        print(f"  Latency: {perf['latency_ms']:.2f} ms")
        print(f"  Memory: {perf['memory_mb']:.2f} MB")
        print(f"  Energy: {perf['energy_mj']:.2f} mJ")
        print()


def demonstrate_track_a_deployment(model, model_name="baseline"):
    """Demonstrate Track A deployment conversion."""
    print("\n" + "="*70)
    print("Track A: Deployment Conversion")
    print("="*70)
    
    # Convert to TFLite
    tflite_path = f"optimized_models/{model_name}.tflite"
    print(f"\nConverting {model_name} to TFLite...")
    convert_to_tflite(model, save_path=tflite_path)
    print(f"✓ Saved to {tflite_path}")
    
    # Deploy to TFLite Micro (generate C array)
    c_path = f"optimized_models/{model_name}_data.cc"
    print(f"\nGenerating TFLite Micro C array...")
    deploy_to_tflite_micro(tflite_path, output_c_path=c_path)
    print(f"✓ Saved to {c_path}")
    
    print(f"\nDeployment artifacts ready for {model_name}!")


def main():
    """Main execution flow."""
    print("\n" + "="*70)
    print("MLS3 Hardware-Aware Optimization Pipeline")
    print("="*70)
    
    # Setup
    ensure_directories()
    
    # Load data
    print("\nLoading CIFAR-10 dataset...")
    train_ds, val_ds, test_ds = load_and_preprocess_data(batch_size=64)
    
    # Load baseline model
    baseline_model = load_baseline_model()
    
    # Create optimized variants
    print("\n⚙ Creating and training optimized variants...")
    print("   (This will take several minutes...)")
    variants = create_and_train_optimized_variants(train_ds, val_ds)
    
    # Apply quantization to one model as example
    print("\n⚙ Applying quantization...")
    apply_quantization(variants["latency"], "latency_optimized", train_ds)
    
    # Benchmark all models
    print("\n⚙ Benchmarking all models...")
    all_results = benchmark_all_models(baseline_model, variants, test_ds)
    
    # Generate comparison report
    generate_comparison_report(all_results["baseline"], all_results)
    
    # Demonstrate Track B (simulation)
    demonstrate_track_b_simulation(baseline_model)
    
    # Demonstrate Track A (deployment)
    demonstrate_track_a_deployment(baseline_model, "baseline")
    
    print("\n" + "="*70)
    print("✓ Pipeline Complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  - optimized_models/latency_optimized.keras")
    print("  - optimized_models/memory_optimized.keras")
    print("  - optimized_models/energy_optimized.keras")
    print("  - optimized_models/*.tflite (quantized models)")
    print("  - optimized_models/*_data.cc (TFLite Micro)")
    print("\nNext steps:")
    print("  1. Review the performance comparison above")
    print("  2. Further tune models based on your target platform")
    print("  3. Deploy to actual hardware for validation")
    print("  4. Generate your analysis report for Part 4")
    print()


if __name__ == "__main__":
    main()
