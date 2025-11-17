from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

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
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluations.",
    )
    parser.add_argument(
        "--baseline-batch-size",
        type=int,
        default=32,
        help="Batch size for baseline training.",
    )
    parser.add_argument(
        "--baseline-epochs",
        type=int,
        default=5,
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
        default=1e-3,
        help="Base learning rate for the baseline schedule.",
    )
    parser.add_argument(
        "--baseline-weight-decay",
        type=float,
        default=1e-4,
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
        default=32,
        help="Batch size for distillation training.",
    )
    parser.add_argument(
        "--report-dir",
        default="reports",
        help="Directory to store JSON summaries.",
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
        "--skip-distillation",
        action="store_true",
        help="Skip distillation experiments.",
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
    baseline_path = Path(args.baseline_path)
    ema_decay = None if args.disable_ema or args.ema_decay >= 1.0 else args.ema_decay
    training_summary = None

    if args.train_baseline or not baseline_path.exists():
        training_summary = train_baseline_model(
            epochs=args.baseline_epochs,
            batch_size=args.baseline_batch_size,
            output_path=str(baseline_path),
            optimizer_name=args.baseline_optimizer,
            base_learning_rate=args.baseline_lr,
            weight_decay=args.baseline_weight_decay,
            ema_decay=ema_decay,
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
    ) = prepare_compression_datasets()
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
        pruner = PruningComparator(base_model_path=str(baseline_path))
        magnitude = pruner.magnitude_based_pruning(target_sparsity=0.6)
        structured = pruner.structured_pruning(target_reduction=0.5)
        pruning_artifacts["magnitude"] = magnitude
        pruning_artifacts["structured"] = structured
        register_model("pruned_magnitude", magnitude["model"], "pruning_magnitude")
        register_model("pruned_structured", structured["model"], "pruning_structured")

    quantization_artifacts: Dict[str, Dict] = {}
    if not args.skip_quantization:
        quant_pipeline = QuantizationPipeline(baseline_model)
        mixed = quant_pipeline.mixed_bit_quantization()
        quantization_artifacts["mixed_bit"] = mixed["mixed_bit_models"]["adaptive_assignment"]
        register_model(
            "mixed_bit",
            mixed["mixed_bit_models"]["adaptive_assignment"]["model"],
            "quantization_mixed",
        )

        ptq_qat = quant_pipeline.post_training_vs_qat_comparison()
        for bits, payload in ptq_qat["ptq_results"].items():
            name = f"ptq_{bits}bit"
            quantization_artifacts[name] = payload
            register_model(name, payload["model"], f"ptq_{bits}")
        for bits, payload in ptq_qat["qat_results"].items():
            name = f"qat_{bits}bit"
            quantization_artifacts[name] = payload
            register_model(name, payload["model"], f"qat_{bits}")

        extreme = quant_pipeline.extreme_quantization()
        for key, payload in extreme.items():
            if isinstance(payload, dict) and "model" in payload:
                quantization_artifacts[key] = payload
                register_model(key, payload["model"], key)

    distillation_artifacts: Dict[str, Dict] = {}
    if not args.skip_distillation:
        student_builder = lambda width=0.5: create_baseline_model(width_multiplier=width, dropout_rate=0.3)
        framework = DistillationFramework(
            teacher_model=baseline_model,
            student_architecture=student_builder,
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
