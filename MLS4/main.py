from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import tf_compat  # noqa: F401  # ensure legacy tf.keras before TensorFlow import
import tensorflow as tf

from application_scenarios import ApplicationScenarioAnalysis
from baseline_model import (
    CALIBRATION_EXPORT_PATH,
    CUSTOM_OBJECTS,
    DATASET_CACHE_PATH,
    DEFAULT_MODEL_PATH,
    create_baseline_model,
    prepare_compression_datasets,
    train_baseline_model,
)
from compression_evaluator import CompressionEvaluator
from part1_pruning import PruningComparator
from part2_quantization import QuantizationPipeline
from part3_distillation import DistillationFramework
from part4_interaction_analysis import CompressionInteractionAnalyzer
from visualization_tools import create_compression_visualizations


AUTOTUNE = tf.data.AUTOTUNE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete model compression pipeline."
    )
    parser.add_argument(
        "--baseline-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained baseline model.",
    )
    parser.add_argument(
        "--train-baseline",
        action="store_true",
        help="Force re-training of the baseline model.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Explicitly skip baseline training even if the baseline model is missing (will error if missing).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for evaluations.",
    )
    parser.add_argument(
        "--baseline-batch-size",
        type=int,
        default=256,
        help="Batch size for baseline training.",
    )
    parser.add_argument(
        "--baseline-epochs",
        type=int,
        default=100,
        help="Total epochs for baseline training.",
    )
    parser.add_argument(
        "--baseline-optimizer",
        choices=("adamw", "sgdw"),
        default="adamw",
        help="Optimizer to use for baseline training.",
    )
    parser.add_argument(
        "--baseline-lr",
        type=float,
        default=2e-4,
        help="Base learning rate for the baseline schedule.",
    )
    parser.add_argument(
        "--baseline-weight-decay",
        type=float,
        default=2e-4,
        help="Weight decay applied to the baseline optimizer.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="EMA decay for baseline training (>=1 disables EMA).",
    )
    parser.add_argument(
        "--disable-ema",
        action="store_true",
        help="Disable EMA tracking even if ema-decay is provided.",
    )
    parser.add_argument(
        "--distillation-batch-size",
        type=int,
        default=64,
        help="Batch size for distillation training.",
    )
    parser.add_argument(
        "--report-dir",
        default="reports",
        help="Directory to store JSON summaries.",
    )
    parser.add_argument(
        "--pruned-output-dir",
        default="results",
        help="Directory to write pruned models (SavedModel/.keras) and TFLite artifacts.",
    )
    # Save pruned Keras models by default to make experiments reproducible.
    # Use --no-save-pruned to disable saving.
    parser.add_argument(
        "--no-save-pruned",
        action="store_false",
        dest="save_pruned",
        default=True,
        help="Disable saving pruned Keras models to --pruned-output-dir.",
    )
    parser.add_argument(
        "--save-pruned-tflite",
        action="store_true",
        help="Also save pruned models as TFLite to --pruned-output-dir when possible.",
    )
    parser.add_argument(
        "--skip-pruning",
        action="store_true",
        help="Skip pruning experiments.",
    )
    parser.add_argument(
        "--skip-quantization",
        action="store_true",
        help="Skip quantization experiments.",
    )
    parser.add_argument(
        "--tflite-platform",
        choices=("generic", "edge_tpu", "arm_cortex_m", "mobile_gpu"),
        default="generic",
        help="Target platform for TFLite quantization.",
    )
    parser.add_argument(
        "--qat-epochs",
        type=int,
        default=10,
        help="Number of epochs for QAT fine-tuning.",
    )
    parser.add_argument(
        "--skip-tflite-quantization",
        action="store_true",
        help="Skip new TFLite quantization (only run legacy quantization).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--skip-distillation",
        action="store_true",
        help="Skip distillation experiments.",
    )
    parser.add_argument(
        "--onlyp",
        action="store_true",
        help="Only run pruning experiments (skip quantization and distillation).",
    )
    parser.add_argument(
        "--onlyq",
        action="store_true",
        help="Only run quantization experiments (skip pruning and distillation).",
    )
    parser.add_argument(
        "--onlyd",
        action="store_true",
        help="Only run distillation experiments (skip pruning and quantization).",
    )
    return parser.parse_args()


def to_dataset(x, y, batch_size):
    """Create TF dataset with proper configuration to avoid sample_weight conflicts."""
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def main() -> None:
    args = parse_args()

    # Validate mutually exclusive options
    only_options = [args.onlyp, args.onlyq, args.onlyd]
    if sum(only_options) > 1:
        raise SystemExit(
            "Error: --onlyp, --onlyq, and --onlyd are mutually exclusive. "
            "Please specify only one."
        )
    
    # Auto-configure skip flags based on --only* options
    if args.onlyp:
        args.skip_quantization = True
        args.skip_distillation = True
        print("Mode: Only Pruning (剪枝专用模式)")
    elif args.onlyq:
        args.skip_pruning = True
        args.skip_distillation = True
        print("Mode: Only Quantization (量化专用模式)")
    elif args.onlyd:
        args.skip_pruning = True
        args.skip_quantization = True
        print("Mode: Only Distillation (蒸馏专用模式)")
    else:
        print("Mode: Full Pipeline (完整流程)")

    # Set random seeds for reproducibility
    seed = args.seed
    tf.keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    print(f"Random seeds set to {seed} for reproducibility")

    baseline_path = Path(args.baseline_path)
    ema_decay = None if args.disable_ema or args.ema_decay >= 1.0 else args.ema_decay
    training_summary = None

    if args.skip_baseline:
        if not baseline_path.exists():
            raise SystemExit(
                f"Baseline model not found at {baseline_path}. Remove --skip-baseline or supply a valid --baseline-path to proceed."
            )
    elif args.train_baseline or not baseline_path.exists():
        training_summary = train_baseline_model(
            epochs=args.baseline_epochs,
            batch_size=args.baseline_batch_size,
            output_path=str(baseline_path),
            optimizer_name=args.baseline_optimizer,
            base_learning_rate=args.baseline_lr,
            weight_decay=args.baseline_weight_decay,
            ema_decay=ema_decay,
            seed=seed,
        )

    baseline_model = tf.keras.models.load_model(baseline_path, custom_objects=CUSTOM_OBJECTS)

    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        calibration_data,
    ) = prepare_compression_datasets(seed=seed)
    train_ds = to_dataset(x_train, y_train, args.batch_size)
    val_ds = to_dataset(x_val, y_val, args.batch_size)
    test_ds = to_dataset(x_test, y_test, args.batch_size)

    baseline_metrics = baseline_model.evaluate(test_ds, verbose=0)
    baseline_accuracy = float(baseline_metrics[1]) if isinstance(baseline_metrics, list) else float(baseline_metrics)
    baseline_size_mb = _estimate_model_size(baseline_model)

    evaluator = CompressionEvaluator(
        baseline_metrics={"test_accuracy": baseline_accuracy, "model_size_mb": baseline_size_mb}
    )

    evaluation_records: List[Dict] = []

    def register_model(model_name: str, model: tf.keras.Model, technique: str) -> None:
        metrics = evaluator.benchmark_compressed_model(
            model=model,
            test_data=test_ds,
            model_name=model_name,
            technique=technique,
        )
        evaluation_records.append(metrics)

    register_model("baseline", baseline_model, "baseline")

    pruning_artifacts: Dict[str, Dict] = {}
    if not args.skip_pruning:
        print("\n" + "="*60)
        print("开始剪枝阶段 (Starting Pruning Stage)")
        print("="*60 + "\n")

        pruner = PruningComparator(base_model_path=str(baseline_path))

        # Magnitude-based pruning with layer-wise strategy and warmup
        print("\n>>> 运行改进的幅度剪枝 (Improved Magnitude Pruning) <<<")
        # Build default save paths for pruned models (kept under results/ for consistency)
        pruned_dir = Path(args.pruned_output_dir)
        pruned_dir.mkdir(parents=True, exist_ok=True)

        magnitude = pruner.magnitude_based_pruning(
            target_sparsity=0.6,
            fine_tune_epochs=15,
            learning_rate=1e-3,
            early_stopping_patience=8,
            use_layer_wise_sparsity=True,
            use_warmup=True,
            save_path=str(pruned_dir / "pruned_magnitude.keras") if args.save_pruned else None,
            save_tflite_path=(str(pruned_dir / "pruned_magnitude.tflite") if args.save_pruned_tflite else None),
        )
        pruning_artifacts["magnitude"] = magnitude
        register_model("pruned_magnitude", magnitude["model"], "pruning_magnitude")

        # Gradual magnitude pruning (multi-stage approach)
        print("\n>>> 运行渐进式剪枝 (Gradual Multi-Stage Pruning) <<<")
        gradual = pruner.gradual_magnitude_pruning(
            target_sparsity=0.6,
            num_stages=3,
            epochs_per_stage=5,
            learning_rate=1e-3,
            use_layer_wise_sparsity=True,
            save_path=str(pruned_dir / "pruned_gradual.keras") if args.save_pruned else None,
            save_tflite_path=(str(pruned_dir / "pruned_gradual.tflite") if args.save_pruned_tflite else None),
        )
        pruning_artifacts["gradual_magnitude"] = gradual
        register_model("pruned_gradual", gradual["model"], "pruning_gradual")

        # Structured pruning with improved training configuration
        print("\n>>> 运行改进的结构化剪枝 (Improved Structured Pruning) <<<")
        structured = pruner.structured_pruning(
            target_reduction=0.5,
            fine_tune_epochs=20,
            learning_rate=1e-3,
            batch_size=128,
            early_stopping_patience=8,
            use_warmup=True,
            use_physical_removal=False,  # Keep false for stability
            save_path=str(pruned_dir / "pruned_structured.keras") if args.save_pruned else None,
            save_tflite_path=(str(pruned_dir / "pruned_structured.tflite") if args.save_pruned_tflite else None),
        )
        pruning_artifacts["structured"] = structured
        register_model("pruned_structured", structured["model"], "pruning_structured")

        print("\n" + "="*60)
        print("剪枝阶段完成 (Pruning Stage Completed)")
        print(f"  幅度剪枝准确率: {magnitude['final_accuracy']:.4f} (稀疏度: {magnitude['sparsity_achieved']:.2%})")
        print(f"  渐进式剪枝准确率: {gradual['final_accuracy']:.4f} (稀疏度: {gradual['sparsity_achieved']:.2%})")
        print(f"  结构化剪枝准确率: {structured['final_accuracy']:.4f} (缩减: {structured['model_size_reduction']:.2%})")
        print("="*60 + "\n")

    quantization_artifacts: Dict[str, Dict] = {}
    if not args.skip_quantization:
        quant_pipeline = QuantizationPipeline(baseline_model)

        # 1. Legacy mixed-bit quantization (kept for compatibility)
        mixed = quant_pipeline.mixed_bit_quantization()
        quantization_artifacts["mixed_bit"] = mixed["mixed_bit_models"]["adaptive_assignment"]
        register_model(
            "mixed_bit",
            mixed["mixed_bit_models"]["adaptive_assignment"]["model"],
            "quantization_mixed",
        )

        # 2. Legacy PTQ vs QAT comparison (kept for compatibility)
        ptq_qat = quant_pipeline.post_training_vs_qat_comparison()
        for bits, payload in ptq_qat["ptq_results"].items():
            name = f"legacy_ptq_{bits}bit"
            quantization_artifacts[name] = payload
            register_model(name, payload["model"], f"legacy_ptq_{bits}")
        for bits, payload in ptq_qat["qat_results"].items():
            name = f"legacy_qat_{bits}bit"
            quantization_artifacts[name] = payload
            register_model(name, payload["model"], f"legacy_qat_{bits}")

        # 3. NEW: Standard TFLite Post-Training Quantization (ASS.md requirement)
        ptq_results = {}
        if not args.skip_tflite_quantization:
            print("Implementing standard TFLite Post-Training Quantization...")
            ptq_results = quant_pipeline.standard_tflite_quantization(
                quantization_type="post_training",
                representative_data=train_ds,
                target_platform=args.tflite_platform
            )
            if "post_training_quantization" in ptq_results and "accuracy" in ptq_results["post_training_quantization"]:
                quantization_artifacts["tflite_ptq"] = ptq_results["post_training_quantization"]
                print(f"TFLite PTQ Accuracy: {ptq_results['post_training_quantization']['accuracy']:.4f}")
                print(f"TFLite PTQ Model Size: {ptq_results['post_training_quantization']['model_size_mb']:.2f} MB")

        # 4. NEW: Standard TFLite Dynamic Range Quantization (ASS.md requirement)
        if not args.skip_tflite_quantization:
            print("Implementing standard TFLite Dynamic Range Quantization...")
            dr_results = quant_pipeline.standard_tflite_quantization(
                quantization_type="dynamic_range",
                target_platform=args.tflite_platform
            )
            if "dynamic_range_quantization" in dr_results and "accuracy" in dr_results["dynamic_range_quantization"]:
                quantization_artifacts["tflite_dynamic_range"] = dr_results["dynamic_range_quantization"]
                print(f"TFLite Dynamic Range Accuracy: {dr_results['dynamic_range_quantization']['accuracy']:.4f}")
                print(f"TFLite Dynamic Range Model Size: {dr_results['dynamic_range_quantization']['model_size_mb']:.2f} MB")

        # 5. NEW: Standard TFLite Quantization-Aware Training (ASS.md requirement)
        if not args.skip_tflite_quantization:
            print("Implementing standard TFLite Quantization-Aware Training...")
            qat_results = quant_pipeline.implement_standard_qat(
                train_dataset=train_ds,
                validation_dataset=val_ds,
                epochs=args.qat_epochs,
                target_platform=args.tflite_platform
            )
            if "tflite_accuracy" in qat_results:
                quantization_artifacts["tflite_qat"] = qat_results
                # Register the QAT Keras model for traditional evaluation
                register_model("tflite_qat_keras", qat_results["qat_keras_model"], "tflite_qat")
                print(f"TFLite QAT Keras Accuracy: {qat_results['keras_accuracy']:.4f}")
                print(f"TFLite QAT TFLite Accuracy: {qat_results['tflite_accuracy']:.4f}")
                print(f"TFLite QAT Training Time: {qat_results['training_time_sec']:.1f}s")

        # 6. NEW: PTQ vs QAT Comprehensive Comparison
        comparison_results = {}
        if not args.skip_tflite_quantization:
            print("Running PTQ vs QAT comprehensive comparison...")
            comparison_results = quant_pipeline.compare_ptq_vs_qat(
                train_dataset=train_ds,
                validation_dataset=val_ds,
                qat_epochs=max(1, args.qat_epochs - 2),  # Slightly shorter for comparison
                target_platform=args.tflite_platform
            )
            if "comparison" in comparison_results:
                quantization_artifacts["ptq_vs_qat_comparison"] = comparison_results
                print(f"PTQ vs QAT Comparison completed")
                if comparison_results.get("comparison"):
                    comp = comparison_results["comparison"]
                    print(f"  PTQ Accuracy: {comp.get('ptq_accuracy', 0):.4f}")
                    print(f"  QAT Accuracy: {comp.get('qat_accuracy', 0):.4f}")
                    print(f"  QAT Gain: {comp.get('accuracy_gain', 0):.4f}")

        # 7. Extreme quantization (kept for compatibility)
        extreme = quant_pipeline.extreme_quantization()
        for key, payload in extreme.items():
            if isinstance(payload, dict) and "model" in payload:
                quantization_artifacts[f"extreme_{key}"] = payload
                register_model(f"extreme_{key}", payload["model"], f"extreme_{key}")

        print("All quantization experiments completed successfully!")

    distillation_artifacts: Dict[str, Dict] = {}
    if not args.skip_distillation:
        student_builder = lambda width=0.5: create_baseline_model(width_multiplier=width, dropout_rate=0.3)
        framework = DistillationFramework(
            teacher_model=baseline_model,
            student_architecture=student_builder,
            batch_size=args.distillation_batch_size,
        )
        temp_search = framework.temperature_optimization(num_trials=3)
        distillation_artifacts["temperature_search"] = temp_search
        register_model(
            "distilled_temp",
            temp_search["best_model"],
            "distillation_temperature",
        )

        progressive = framework.progressive_distillation()
        if progressive.get("final_student") is not None:
            distillation_artifacts["progressive"] = progressive
            register_model(
                "distilled_progressive",
                progressive["final_student"],
                "distillation_progressive",
            )

        attention = framework.attention_transfer()
        distillation_artifacts["attention"] = attention
        register_model(
            "distilled_attention",
            attention["combined_distillation_results"]["model"],
            "distillation_attention",
        )

        feature = framework.feature_matching_distillation()
        distillation_artifacts["feature"] = feature
        register_model(
            "distilled_feature",
            feature["student_model"],
            "distillation_feature",
        )

    analyzer = CompressionInteractionAnalyzer(baseline_model, test_ds)
    combinations = analyzer.comprehensive_compression_analysis(
        pruning_models=pruning_artifacts,
        quantized_models=quantization_artifacts,
        distilled_models=distillation_artifacts,
        student_builder=lambda width=0.45: create_baseline_model(width_multiplier=width, dropout_rate=0.3),
    )
    pareto = analyzer.pareto_frontier_analysis(evaluation_records)
    pipeline_opts = analyzer.compression_pipeline_optimization(evaluation_records)
    failures = analyzer.failure_mode_analysis(evaluation_records)

    create_compression_visualizations(evaluation_records)

    scenario_analysis = ApplicationScenarioAnalysis(evaluation_records)
    scenario_results = {
        "mobile": scenario_analysis.mobile_deployment_optimization(),
        "edge": scenario_analysis.edge_device_optimization(),
        "cloud": scenario_analysis.cloud_inference_optimization(),
    }

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "pipeline_evaluations.json", "w", encoding="utf-8") as fh:
        json.dump(evaluation_records, fh, indent=2)
    with open(results_dir / "calibration_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "calibration_size": int(len(calibration_data[0])),
                "dataset_cache": str(DATASET_CACHE_PATH),
                "calibration_file": str(CALIBRATION_EXPORT_PATH),
            },
            fh,
            indent=2,
        )
    summary = {
        "baseline_accuracy": baseline_accuracy,
        "combinations": combinations,
        "pareto": pareto,
        "pipeline_optimization": pipeline_opts,
        "failures": failures,
        "scenarios": scenario_results,
        "evaluations": evaluation_records,
        "baseline_training": training_summary,
        "dataset_cache": str(DATASET_CACHE_PATH),
        "calibration_file": str(CALIBRATION_EXPORT_PATH),
    }
    with open(report_dir / "pipeline_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Summary written to {report_dir / 'pipeline_summary.json'}")


def _estimate_model_size(model: tf.keras.Model) -> float:
    total_bytes = sum(weight.numpy().nbytes for weight in model.weights)
    return total_bytes / (1024 * 1024)


if __name__ == "__main__":
    main()
