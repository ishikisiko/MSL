from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tf_compat  # noqa: F401  # ensure TensorFlow uses legacy keras API
import tensorflow as tf

from part2_quantization import QuantizationPipeline


@dataclass
class CombinationResult:
    name: str
    accuracy: float
    compression_ratio: float
    notes: str
    model: tf.keras.Model


class CompressionInteractionAnalyzer:
    """
    Study how pruning, quantization, and distillation interact with each other.

    The analyzer consumes ready-made artifacts (models or dictionaries containing
    models) and composes them into new hybrids so we can reason about ordering,
    Pareto fronts, and failure modes.
    """

    def __init__(
        self,
        baseline_model: tf.keras.Model,
        evaluation_dataset: tf.data.Dataset,
    ) -> None:
        self.baseline_model = baseline_model
        self.test_dataset = evaluation_dataset
        self.interaction_results: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Combination experiments
    # ------------------------------------------------------------------ #
    def comprehensive_compression_analysis(
        self,
        pruning_models: Optional[Dict[str, Any]] = None,
        quantized_models: Optional[Dict[str, Any]] = None,
        distilled_models: Optional[Dict[str, Any]] = None,
        student_builder: Optional[Any] = None,
        target_bits: int = 8,
        target_sparsity: float = 0.5,
    ) -> Dict[str, Dict]:
        """
        Evaluate key combinations across techniques and ordering.
        """

        pruning_models = pruning_models or {}
        quantized_models = quantized_models or {}
        distilled_models = distilled_models or {}

        combos: Dict[str, Dict] = {
            "pruning_then_quantization": {},
            "quantization_then_pruning": {},
            "pruning_with_distillation": {},
            "quantization_with_distillation": {},
            "all_three_combined": {},
            "ordering_sensitivity_analysis": {},
        }

        best_pruned = self._select_best_model(pruning_models)
        best_quantized = self._select_best_model(quantized_models)

        if best_pruned is not None:
            pq_model = QuantizationPipeline.quantize_model(best_pruned, bits=target_bits)
            combos["pruning_then_quantization"] = self._evaluate_combination(
                "prune->quant", pq_model, notes="Magnitude pruning followed by PTQ."
            )

        if best_quantized is not None:
            qp_model = self._apply_pruning(best_quantized, target_sparsity)
            combos["quantization_then_pruning"] = self._evaluate_combination(
                "quant->prune", qp_model, notes="Uniform PTQ before global pruning."
            )

        if best_pruned is not None and student_builder is not None:
            from part3_distillation import DistillationFramework

            framework = DistillationFramework(
                teacher_model=best_pruned,
                student_architecture=student_builder,
                cache_datasets=False,
            )
            dist_results = framework.temperature_optimization(num_trials=1, width_multiplier=0.5)
            student = dist_results["knowledge_transfer_metrics"][0]["student_model"]
            combos["pruning_with_distillation"] = self._evaluate_combination(
                "prune+distill",
                student,
                notes="Distill small student from pruned teacher.",
            )

        if best_quantized is not None and student_builder is not None:
            from part3_distillation import DistillationFramework

            framework = DistillationFramework(
                teacher_model=best_quantized,
                student_architecture=student_builder,
                cache_datasets=False,
            )
            dist_results = framework.temperature_optimization(num_trials=1, width_multiplier=0.5)
            student = dist_results["knowledge_transfer_metrics"][0]["student_model"]
            combos["quantization_with_distillation"] = self._evaluate_combination(
                "quant+distill",
                student,
                notes="Distill after PTQ.",
            )

        if (
            best_pruned is not None
            and best_quantized is not None
            and student_builder is not None
        ):
            pq_model = QuantizationPipeline.quantize_model(best_pruned, bits=target_bits)
            combo_model = self._apply_pruning(pq_model, target_sparsity)
            from part3_distillation import DistillationFramework

            framework = DistillationFramework(
                teacher_model=combo_model,
                student_architecture=student_builder,
                cache_datasets=False,
            )
            dist_results = framework.temperature_optimization(num_trials=1, width_multiplier=0.4)
            student = dist_results["knowledge_transfer_metrics"][0]["student_model"]
            combos["all_three_combined"] = self._evaluate_combination(
                "prune->quant->distill",
                student,
                notes="Sequential pipeline covering all three techniques.",
            )

        combos["ordering_sensitivity_analysis"] = self._ordering_sensitivity(
            combos["pruning_then_quantization"],
            combos["quantization_then_pruning"],
        )

        self.interaction_results["combinations"] = combos
        return combos

    # ------------------------------------------------------------------ #
    # Pareto / optimization analysis
    # ------------------------------------------------------------------ #
    def pareto_frontier_analysis(
        self,
        evaluation_metrics: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compute Pareto frontiers for accuracy vs. size and latency.
        """

        if not evaluation_metrics:
            return {}

        size_points = [
            (metric.get("compression_ratio", 1.0), metric.get("test_accuracy", 0.0), metric.get("model_name", ""))
            for metric in evaluation_metrics
            if "compression_ratio" in metric and "test_accuracy" in metric
        ]
        latency_points = [
            (metric.get("single_inference_ms", 0.0), metric.get("test_accuracy", 0.0), metric.get("model_name", ""))
            for metric in evaluation_metrics
            if "single_inference_ms" in metric and "test_accuracy" in metric
        ]

        size_front = self._compute_pareto(size_points, maximize_first=True)
        latency_front = self._compute_pareto(latency_points, maximize_first=False)

        recommendations = {
            "mobile": self._recommend_model(size_front, prefer="compression"),
            "edge": self._recommend_model(latency_front, prefer="latency"),
            "cloud": self._recommend_model(size_front + latency_front, prefer="accuracy"),
        }

        analysis = {
            "accuracy_vs_size_frontier": size_front,
            "accuracy_vs_latency_frontier": latency_front,
            "non_dominated_solutions": size_front + latency_front,
            "use_case_recommendations": recommendations,
        }
        self.interaction_results["pareto"] = analysis
        return analysis

    def compression_pipeline_optimization(
        self,
        evaluation_metrics: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Highlight promising pipelines based on collected metrics.
        """

        if not evaluation_metrics:
            return {}

        sorted_by_accuracy = sorted(
            evaluation_metrics, key=lambda item: item.get("test_accuracy", 0.0), reverse=True
        )
        high_accuracy = sorted_by_accuracy[:3]

        sorted_by_ratio = sorted(
            evaluation_metrics, key=lambda item: item.get("compression_ratio", 1.0), reverse=True
        )
        high_compression = sorted_by_ratio[:3]

        fastest = sorted(
            evaluation_metrics, key=lambda item: item.get("single_inference_ms", float("inf"))
        )[:3]

        analysis = {
            "accuracy_first": high_accuracy,
            "size_first": high_compression,
            "latency_first": fastest,
        }
        self.interaction_results["pipeline_optimization"] = analysis
        return analysis

    def failure_mode_analysis(
        self,
        evaluation_metrics: Sequence[Dict[str, Any]],
        accuracy_floor: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Identify compression settings that lead to severe accuracy drops.
        """

        failing = [
            metric
            for metric in evaluation_metrics
            if metric.get("test_accuracy", 0.0) < accuracy_floor
        ]

        if not failing:
            analysis = {"failure_cases": [], "root_causes": {}, "guidelines": []}
            self.interaction_results["failures"] = analysis
            return analysis

        root_causes: Dict[str, int] = {}
        for case in failing:
            key = case.get("technique", "unknown")
            root_causes[key] = root_causes.get(key, 0) + 1

        guidelines = [
            "Avoid combining extreme sparsity with binary quantization.",
            "Run QAT when compression ratio exceeds 10x.",
            "Use progressive distillation if PTQ accuracy drops >5%.",
        ]

        analysis = {
            "failure_cases": failing,
            "root_causes": root_causes,
            "guidelines": guidelines,
        }
        self.interaction_results["failures"] = analysis
        return analysis

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _select_best_model(self, artifacts: Dict[str, Any]) -> Optional[tf.keras.Model]:
        best_model = None
        best_accuracy = -np.inf
        for payload in artifacts.values():
            model = self._extract_model(payload)
            if model is None:
                continue
            accuracy = payload.get("accuracy") if isinstance(payload, dict) else None
            if accuracy is None:
                accuracy = float(model.evaluate(self.test_dataset, verbose=0)[1])  # type: ignore[index]
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
        return best_model

    def _extract_model(self, payload: Any) -> Optional[tf.keras.Model]:
        if payload is None:
            return None
        if isinstance(payload, tf.keras.Model):
            return payload
        if isinstance(payload, dict):
            model = payload.get("model")
            if isinstance(model, tf.keras.Model):
                return model
        return None

    def _evaluate_combination(
        self,
        name: str,
        model: tf.keras.Model,
        notes: str,
    ) -> Dict:
        metrics = model.evaluate(self.test_dataset, verbose=0)
        accuracy = float(metrics[1]) if isinstance(metrics, list) and len(metrics) > 1 else float(metrics)
        size_bits = sum(w.numpy().size * 32 for w in model.trainable_weights)
        compression_ratio = (
            sum(w.numpy().size * 32 for w in self.baseline_model.trainable_weights)
            / max(1, size_bits)
        )
        result = CombinationResult(
            name=name,
            accuracy=accuracy,
            compression_ratio=compression_ratio,
            notes=notes,
            model=model,
        )
        return result.__dict__

    def _apply_pruning(
        self,
        model: tf.keras.Model,
        target_sparsity: float,
    ) -> tf.keras.Model:
        clone = tf.keras.models.clone_model(model)
        clone.set_weights(model.get_weights())
        all_weights = np.concatenate(
            [weight.flatten() for layer in clone.layers for weight in layer.get_weights() if weight.size]
        )
        if all_weights.size == 0:
            return clone
        threshold = np.percentile(np.abs(all_weights), target_sparsity * 100)
        for layer in clone.layers:
            weights = layer.get_weights()
            if not weights:
                continue
            pruned = []
            for weight in weights:
                mask = np.abs(weight) >= threshold
                pruned.append(weight * mask)
            layer.set_weights(pruned)
        return clone

    def _ordering_sensitivity(
        self,
        first_combo: Dict,
        second_combo: Dict,
    ) -> Dict[str, Any]:
        if not first_combo or not second_combo:
            return {}
        accuracy_delta = first_combo["accuracy"] - second_combo["accuracy"]
        compression_delta = (
            first_combo["compression_ratio"] - second_combo["compression_ratio"]
        )
        return {
            "accuracy_delta": accuracy_delta,
            "compression_delta": compression_delta,
            "preferred_order": "prune->quant"
            if accuracy_delta >= 0
            else "quant->prune",
        }

    def _compute_pareto(
        self,
        points: Sequence[Tuple[float, float, str]],
        maximize_first: bool,
    ) -> List[Dict[str, Any]]:
        if not points:
            return []
        sorted_points = sorted(
            points,
            key=lambda item: item[0],
            reverse=maximize_first,
        )
        frontier = []
        best_second = -np.inf
        for first, second, name in sorted_points:
            if second > best_second:
                frontier.append({"model_name": name, "x": first, "y": second})
                best_second = second
        return frontier

    def _recommend_model(
        self,
        frontier: Sequence[Dict[str, Any]],
        prefer: str,
    ) -> Optional[Dict[str, Any]]:
        if not frontier:
            return None
        if prefer == "compression":
            return max(frontier, key=lambda item: item["x"])
        if prefer == "latency":
            return min(frontier, key=lambda item: item["x"])
        return max(frontier, key=lambda item: item["y"])
