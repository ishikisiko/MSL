from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def create_compression_visualizations(
    evaluation_results: Iterable[Dict],
    output_dir: str = "visualizations",
) -> None:
    """
    Generate comprehensive analysis visualizations.
    """

    results = list(evaluation_results)
    if not results:
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _plot_accuracy_vs_compression(results, output_path / "accuracy_vs_compression.png")
    _plot_pareto_heatmap(results, output_path / "pareto_heatmap.png")
    _plot_training_convergence(results, output_path / "training_convergence.png")
    _plot_interaction_matrix(results, output_path / "interaction_matrix.png")
    _plot_radar_chart(results, output_path / "tradeoff_radar.png")


def _plot_accuracy_vs_compression(results: List[Dict], path: Path) -> None:
    filtered = [
        res
        for res in results
        if "compression_ratio" in res and "test_accuracy" in res
    ]
    if not filtered:
        return

    plt.figure(figsize=(6, 4))
    for entry in filtered:
        plt.scatter(
            entry["compression_ratio"],
            entry["test_accuracy"],
            label=entry.get("model_name", "model"),
        )
    plt.xlabel("Compression Ratio (x)")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.3)
    if len(filtered) <= 10:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_pareto_heatmap(results: List[Dict], path: Path) -> None:
    filtered = [
        res
        for res in results
        if "single_inference_ms" in res and "test_accuracy" in res
    ]
    if not filtered:
        return
    latencies = np.array([res["single_inference_ms"] for res in filtered])
    accuracies = np.array([res["test_accuracy"] for res in filtered])
    heatmap = np.outer(accuracies, 1 / np.maximum(latencies, 1e-3))
    plt.figure(figsize=(5, 4))
    sns.heatmap(heatmap, cmap="viridis")
    plt.title("Accuracy vs Latency Interaction")
    plt.xlabel("Models (latency rank)")
    plt.ylabel("Models (accuracy rank)")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_training_convergence(results: List[Dict], path: Path) -> None:
    histories = [
        res.get("training_history")
        for res in results
        if isinstance(res.get("training_history"), dict)
    ]
    if not histories:
        return
    plt.figure(figsize=(6, 4))
    for history in histories[:5]:
        if "loss" in history:
            plt.plot(history["loss"], alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Convergence (first 5 models)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_interaction_matrix(results: List[Dict], path: Path) -> None:
    techniques = ["pruning", "quantization", "distillation"]
    matrix = np.zeros((len(techniques), len(techniques)))
    baseline_accuracy = max(res.get("test_accuracy", 0.0) for res in results) or 1.0
    for i, ti in enumerate(techniques):
        for j, tj in enumerate(techniques):
            combined = [
                res
                for res in results
                if ti in res.get("technique", "") and tj in res.get("technique", "")
            ]
            if combined:
                accuracy = np.mean([item.get("test_accuracy", 0.0) for item in combined])
                matrix[i, j] = accuracy / baseline_accuracy
            else:
                matrix[i, j] = 0.0
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        xticklabels=techniques,
        yticklabels=techniques,
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.1,
    )
    plt.title("Technique Interaction Matrix (normalized accuracy)")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_radar_chart(results: List[Dict], path: Path) -> None:
    if not results:
        return
    best = max(results, key=lambda res: res.get("test_accuracy", 0.0))
    stats = {
        "accuracy": best.get("test_accuracy", 0.0),
        "compression": best.get("compression_ratio", 1.0),
        "latency": 1 / max(1.0, best.get("single_inference_ms", 1.0)),
        "memory": 1 / max(1.0, best.get("peak_memory_mb", 1.0)),
        "energy": 1 / max(1.0, best.get("energy_consumption", 1.0)),
    }
    labels = list(stats.keys())
    values = list(stats.values())
    values.append(values[0])
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles.append(angles[0])
    plt.figure(figsize=(5, 5))
    plt.polar(angles, values, "o-", linewidth=2)
    plt.fill(angles, values, alpha=0.25)
    plt.xticks(angles[:-1], labels)
    plt.title(f"Trade-off Radar: {best.get('model_name', 'best_model')}")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
