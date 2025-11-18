import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tf_compat  # noqa: F401  # force tf.keras legacy mode for tfmot pruning
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from baseline_model import CUSTOM_OBJECTS, prepare_compression_datasets


DatasetLike = Union[
    tf.data.Dataset,
    Tuple[np.ndarray, np.ndarray],
]


PRUNABLE_LAYER_TYPES = (
    tf.keras.layers.Conv2D,
    tf.keras.layers.DepthwiseConv2D,
    tf.keras.layers.SeparableConv2D,
    tf.keras.layers.Dense,
)


@dataclass
class DatasetBundle:
    train: tf.data.Dataset
    val: tf.data.Dataset
    test: tf.data.Dataset
    train_size: int
    val_size: int
    test_size: int


class PruningComparator:
    """Utility class that centralizes all pruning experiments for Part I."""

    def __init__(
        self,
        base_model_path: str,
        cache_datasets: bool = True,
    ) -> None:
        self.base_model = tf.keras.models.load_model(base_model_path, custom_objects=CUSTOM_OBJECTS)
        self.pruning_results: Dict[str, Dict] = {}
        self._dataset_bundle: Optional[DatasetBundle] = None
        self._cached_batch_size: Optional[int] = None
        self.cache_datasets = cache_datasets

        # Serialize optimizer / loss / metrics so clones can be compiled identically.
        self._optimizer_config = (
            tf.keras.optimizers.serialize(self.base_model.optimizer)
            if hasattr(self.base_model, "optimizer") and self.base_model.optimizer
            else tf.keras.optimizers.serialize(
                tf.keras.optimizers.Adam(learning_rate=1e-3)
            )
        )
        loss_obj = (
            self.base_model.loss
            if getattr(self.base_model, "loss", None)
            else tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        )
        self._loss_config = tf.keras.losses.serialize(loss_obj)
        if getattr(self.base_model, "metrics", None):
            metric_configs = []
            for metric in self.base_model.metrics:
                try:
                    metric_configs.append(tf.keras.metrics.serialize(metric))
                    continue
                except Exception:
                    identifier = getattr(metric, "name", None) or getattr(metric, "_name", None)
                    if identifier:
                        metric_configs.append(identifier)
                        continue
                metric_configs.append(
                    tf.keras.metrics.serialize(
                        tf.keras.metrics.CategoricalAccuracy(name="accuracy")
                    )
                )
            self._metric_configs = metric_configs
        else:
            self._metric_configs = [
                tf.keras.metrics.serialize(
                    tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
                )
            ]

        if cache_datasets:
            self._dataset_bundle = self._prepare_default_datasets()
            self._cached_batch_size = 128

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #
    def magnitude_based_pruning(
        self,
        target_sparsity: float = 0.7,
        pruning_schedule: str = "polynomial",
        train_data: Optional[DatasetLike] = None,
        val_data: Optional[DatasetLike] = None,
        fine_tune_epochs: int = 5,
        batch_size: int = 128,
        learning_rate: float = 1e-4,
        early_stopping_patience: int = 3,
        save_path: Optional[str] = None,
        save_tflite_path: Optional[str] = None,
    ) -> Dict:
        """Perform magnitude-based pruning with configurable schedules."""

        print(f"\n{'='*60}")
        print(f"开始幅度剪枝 (Starting Magnitude-Based Pruning)")
        print(f"目标稀疏度: {target_sparsity:.2%} | 剪枝计划: {pruning_schedule}")
        print(f"{'='*60}\n")

        schedule = self._build_pruning_schedule(
            schedule_name=pruning_schedule,
            target_sparsity=target_sparsity,
            fine_tune_epochs=fine_tune_epochs,
            batch_size=batch_size,
            train_size=self._resolve_dataset_size(train_data, default="train"),
        )

        cloned_model = self._clone_and_compile(learning_rate=learning_rate)
        pruned_model = self._apply_pruning_wrappers(
            cloned_model,
            pruning_schedule=schedule,
        )

        # The pruning wrapper returns a new model which requires explicit
        # compilation before training. Reuse optimizer/loss/metrics from the
        # cloned (compiled) model but allow learning rate overrides.
        optimizer = tf.keras.optimizers.deserialize(self._optimizer_config)
        self._safe_set_learning_rate(optimizer, learning_rate)

        loss = tf.keras.losses.deserialize(self._loss_config, custom_objects=CUSTOM_OBJECTS)

        metrics = []
        for metric_config in self._metric_configs:
            if isinstance(metric_config, str):
                metrics.append(metric_config)
            else:
                try:
                    metrics.append(tf.keras.metrics.deserialize(metric_config, custom_objects=CUSTOM_OBJECTS))
                except Exception:
                    metrics.append(tf.keras.metrics.CategoricalAccuracy(name="accuracy"))

        pruned_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        train_ds = self._resolve_dataset(
            dataset=train_data,
            fallback_split="train",
            batch_size=batch_size,
            augment=True,
            shuffle=True,
        )
        val_ds = self._resolve_dataset(
            dataset=val_data,
            fallback_split="val",
            batch_size=batch_size,
        )

        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=early_stopping_patience,
                restore_best_weights=True,
            ),
        ]

        history = pruned_model.fit(
            train_ds,
            epochs=fine_tune_epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=2,
        )

        stripped_model = self._strip_and_compile(pruned_model)
        final_accuracy = self._evaluate_accuracy(stripped_model, val_ds)
        layer_sparsity = self._compute_layer_sparsity(stripped_model)

        results = {
            "model": stripped_model,
            "final_accuracy": final_accuracy,
            "sparsity_achieved": np.mean(
                [layer["sparsity"] for layer in layer_sparsity.values()]
            )
            if layer_sparsity
            else 0.0,
            "layer_sparsity_analysis": layer_sparsity,
            "training_history": history.history,
        }

        self.pruning_results["magnitude_based"] = results

        # Optional: persist model to disk (SavedModel / HDF5)
        if save_path:
            try:
                results["model"].save(save_path)
            except Exception as exc:  # pragma: no cover - IO operations
                print(f"Failed to save pruned model to {save_path}: {exc}")

        # Optional: persist a TFLite representation of the pruned model
        if save_tflite_path:
            try:
                converter = tf.lite.TFLiteConverter.from_keras_model(results["model"])
                tflite_model = converter.convert()
                with open(save_tflite_path, "wb") as f:
                    f.write(tflite_model)
            except Exception as exc:  # pragma: no cover - IO/conversion may fail locally
                print(f"Failed to convert and save TFLite model to {save_tflite_path}: {exc}")

        print(f"\n{'='*60}")
        print(f"幅度剪枝完成 (Magnitude-Based Pruning Completed)")
        print(f"最终准确率: {results['final_accuracy']:.4f} | 稀疏度: {results['sparsity_achieved']:.2%}")
        print(f"{'='*60}\n")

        return results

    def structured_pruning(
        self,
        target_reduction: float = 0.5,
        importance_metric: str = "l1_norm",
        fine_tune_epochs: int = 10,
        learning_rate: float = 5e-4,
        batch_size: int = 32,
        train_data: Optional[DatasetLike] = None,
        val_data: Optional[DatasetLike] = None,
        save_path: Optional[str] = None,
        save_tflite_path: Optional[str] = None,
    ) -> Dict:
        """Apply coarse-grained filter/channel pruning."""

        print(f"\n{'='*60}")
        print(f"开始结构化剪枝 (Starting Structured Pruning)")
        print(f"目标缩减比例: {target_reduction:.2%} | 重要性度量: {importance_metric}")
        print(f"{'='*60}\n")

        model = self._clone_and_compile(learning_rate=learning_rate)
        conv_masks: Dict[str, np.ndarray] = {}
        filters_removed: Dict[str, List[int]] = {}
        architecture_changes: Dict[str, str] = {}

        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                weights = layer.get_weights()
                if not weights:
                    continue

                kernel = weights[0]
                bias = weights[1] if len(weights) > 1 else None
                num_filters = kernel.shape[-1]
                if num_filters <= 1:
                    continue
                remove_count = int(num_filters * target_reduction)
                remove_count = max(1, remove_count)
                if remove_count >= num_filters:
                    remove_count = num_filters - 1
                importance = self._compute_filter_importance(
                    kernel,
                    metric=importance_metric,
                )
                remove_indices = np.argsort(importance)[:remove_count]
                mask = np.ones(num_filters, dtype=bool)
                mask[remove_indices] = False
                conv_masks[layer.name] = mask
                filters_removed[layer.name] = remove_indices.tolist()
                architecture_changes[layer.name] = (
                    f"Zeroed {remove_count}/{num_filters} filters "
                    f"({target_reduction:.2%} target)"
                )

                kernel[..., ~mask] = 0.0
                if bias is not None:
                    bias[~mask] = 0.0
                layer.set_weights([kernel, bias] if bias is not None else [kernel])

            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                # Align BN parameters with preceding convolution mask.
                prev_mask = conv_masks.get(self._find_previous_conv(layer, model))
                if prev_mask is None:
                    continue
                gamma, beta, moving_mean, moving_var = layer.get_weights()
                gamma[~prev_mask] = 0.0
                beta[~prev_mask] = 0.0
                moving_mean[~prev_mask] = 0.0
                moving_var[~prev_mask] = 1.0
                layer.set_weights([gamma, beta, moving_mean, moving_var])

        train_ds = self._resolve_dataset(
            dataset=train_data,
            fallback_split="train",
            batch_size=batch_size,
            augment=True,
            shuffle=True,
        )
        val_ds = self._resolve_dataset(
            dataset=val_data,
            fallback_split="val",
            batch_size=batch_size,
        )

        history = model.fit(
            train_ds,
            epochs=fine_tune_epochs,
            validation_data=val_ds,
            verbose=2,
        )

        final_accuracy = self._evaluate_accuracy(model, val_ds)
        total_filters = sum(mask.size for mask in conv_masks.values())
        removed_filters = sum((~mask).sum() for mask in conv_masks.values())
        size_reduction = (
            removed_filters / total_filters if total_filters else 0.0
        )

        results = {
            "model": model,
            "filters_removed_per_layer": filters_removed,
            "architecture_changes": architecture_changes,
            "final_accuracy": final_accuracy,
            "model_size_reduction": size_reduction,
            "training_history": history.history,
        }
        self.pruning_results["structured"] = results

        # Optional: persist model to disk (SavedModel / HDF5)
        if save_path:
            try:
                results["model"].save(save_path)
            except Exception as exc:  # pragma: no cover - IO operations
                print(f"Failed to save structured pruned model to {save_path}: {exc}")

        # Optional: persist a TFLite representation of the pruned model
        if save_tflite_path:
            try:
                converter = tf.lite.TFLiteConverter.from_keras_model(results["model"])
                tflite_model = converter.convert()
                with open(save_tflite_path, "wb") as f:
                    f.write(tflite_model)
            except Exception as exc:  # pragma: no cover - IO/conversion may fail locally
                print(f"Failed to convert and save TFLite model to {save_tflite_path}: {exc}")

        print(f"\n{'='*60}")
        print(f"结构化剪枝完成 (Structured Pruning Completed)")
        print(f"最终准确率: {results['final_accuracy']:.4f} | 模型大小缩减: {results['model_size_reduction']:.2%}")
        print(f"{'='*60}\n")

        return results

    def gradual_vs_oneshot_pruning(
        self,
        target_sparsity: float = 0.7,
        fine_tune_epochs: int = 6,
        batch_size: int = 128,
    ) -> Dict:
        """Compare gradual polynomial pruning vs one-shot constant pruning."""

        print(f"\n{'='*60}")
        print(f"开始渐进式与一次性剪枝对比 (Starting Gradual vs One-shot Pruning Comparison)")
        print(f"目标稀疏度: {target_sparsity:.2%}")
        print(f"{'='*60}\n")

        gradual = self.magnitude_based_pruning(
            target_sparsity=target_sparsity,
            pruning_schedule="gradual",
            fine_tune_epochs=fine_tune_epochs,
            batch_size=batch_size,
        )
        oneshot = self.magnitude_based_pruning(
            target_sparsity=target_sparsity,
            pruning_schedule="constant",
            fine_tune_epochs=max(1, fine_tune_epochs // 2),
            batch_size=batch_size,
        )

        stability = {
            "accuracy_delta": gradual["final_accuracy"] - oneshot["final_accuracy"],
            "sparsity_delta": gradual["sparsity_achieved"]
            - oneshot["sparsity_achieved"],
        }
        convergence = {
            "gradual_epochs": len(gradual["training_history"].get("loss", [])),
            "oneshot_epochs": len(oneshot["training_history"].get("loss", [])),
            "gradual_history": gradual["training_history"],
            "oneshot_history": oneshot["training_history"],
        }

        comparison = {
            "gradual_pruning": gradual,
            "oneshot_pruning": oneshot,
            "convergence_analysis": convergence,
            "stability_metrics": stability,
        }
        self.pruning_results["gradual_vs_oneshot"] = comparison

        print(f"\n{'='*60}")
        print(f"渐进式与一次性剪枝对比完成 (Gradual vs One-shot Pruning Comparison Completed)")
        print(f"渐进式准确率: {comparison['gradual_pruning']['final_accuracy']:.4f}")
        print(f"一次性准确率: {comparison['oneshot_pruning']['final_accuracy']:.4f}")
        print(f"准确率差异: {comparison['stability_metrics']['accuracy_delta']:.4f}")
        print(f"{'='*60}\n")

        return comparison

    def analyze_pruning_sensitivity(
        self,
        probe_sparsity: float = 0.3,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        """Individually prune each layer to measure accuracy drop."""

        print(f"\n{'='*60}")
        print(f"开始剪枝敏感性分析 (Starting Pruning Sensitivity Analysis)")
        print(f"探查稀疏度: {probe_sparsity:.2%}")
        print(f"{'='*60}\n")

        val_ds = self._resolve_dataset(
            dataset=None,
            fallback_split="val",
            batch_size=batch_size,
        )
        baseline_accuracy = self._evaluate_accuracy(self.base_model, val_ds)

        sensitivity: Dict[str, float] = {}
        for layer in self.base_model.layers:
            if not layer.get_weights():
                continue

            model_clone = self._clone_and_compile()
            target_layer = model_clone.get_layer(layer.name)
            weights = target_layer.get_weights()
            pruned_weights = []
            for weight in weights:
                flat = weight.flatten()
                kth = int(len(flat) * probe_sparsity)
                if kth <= 0:
                    pruned_weights.append(weight)
                    continue
                kth = min(len(flat) - 1, kth)
                threshold = np.partition(np.abs(flat), kth)[kth]
                mask = np.abs(weight) >= threshold
                pruned_weights.append(weight * mask)
            target_layer.set_weights(pruned_weights)
            acc = self._evaluate_accuracy(model_clone, val_ds)
            sensitivity[layer.name] = float(baseline_accuracy - acc)

        self.pruning_results["sensitivity"] = sensitivity
        return sensitivity

    def lottery_ticket_hypothesis_test(
        self,
        iterations: int = 3,
        initial_sparsity: float = 0.3,
        sparsity_increment: float = 0.2,
        fine_tune_epochs: int = 3,
        batch_size: int = 128,
        learning_rate: float = 1e-4,
    ) -> Dict:
        """Run iterative magnitude pruning with weight rewinding."""

        train_ds = self._resolve_dataset(
            dataset=None,
            fallback_split="train",
            batch_size=batch_size,
            augment=True,
            shuffle=True,
        )
        val_ds = self._resolve_dataset(
            dataset=None,
            fallback_split="val",
            batch_size=batch_size,
        )

        initial_weights = self.base_model.get_weights()
        masks = [np.ones_like(weight, dtype=bool) for weight in initial_weights]

        iteration_results = []
        target_sparsity = initial_sparsity

        for iteration in range(iterations):
            starting_weights = [
                weight * mask.astype(weight.dtype)
                for weight, mask in zip(initial_weights, masks)
            ]
            model = self._clone_and_compile(learning_rate=learning_rate)
            model.set_weights(starting_weights)

            schedule = self._build_pruning_schedule(
                schedule_name="polynomial",
                target_sparsity=target_sparsity,
                fine_tune_epochs=fine_tune_epochs,
                batch_size=batch_size,
                train_size=self._resolve_dataset_size(None, default="train"),
            )
            pruned_model = self._apply_pruning_wrappers(
                model,
                pruning_schedule=schedule,
            )

            # Ensure the pruning-wrapped model is compiled for training
            optimizer = tf.keras.optimizers.deserialize(self._optimizer_config)
            self._safe_set_learning_rate(optimizer, learning_rate)

            loss = tf.keras.losses.deserialize(self._loss_config, custom_objects=CUSTOM_OBJECTS)

            metrics = []
            for metric_config in self._metric_configs:
                if isinstance(metric_config, str):
                    metrics.append(metric_config)
                else:
                    try:
                        metrics.append(tf.keras.metrics.deserialize(metric_config, custom_objects=CUSTOM_OBJECTS))
                    except Exception:
                        metrics.append(tf.keras.metrics.CategoricalAccuracy(name="accuracy"))

            pruned_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            history = pruned_model.fit(
                train_ds,
                epochs=fine_tune_epochs,
                validation_data=val_ds,
                callbacks=[tfmot.sparsity.keras.UpdatePruningStep()],
                verbose=2,
            )

            stripped = self._strip_and_compile(pruned_model)
            accuracy = self._evaluate_accuracy(stripped, val_ds)

            new_masks = []
            for weight in stripped.get_weights():
                new_masks.append(np.abs(weight) > 1e-9)
            masks = [
                np.logical_and(old_mask, new_mask)
                for old_mask, new_mask in zip(masks, new_masks)
            ]

            iteration_results.append(
                {
                    "iteration": iteration + 1,
                    "target_sparsity": target_sparsity,
                    "final_accuracy": accuracy,
                    "history": history.history,
                }
            )

            target_sparsity = min(0.99, target_sparsity + sparsity_increment)

        results = {
            "iterations": iteration_results,
            "winning_ticket_masks": masks,
        }
        self.pruning_results["lottery_ticket"] = results
        return results

    # ---------------------------------------------------------------------- #
    # Internal utilities
    # ---------------------------------------------------------------------- #
    def _apply_pruning_wrappers(
        self,
        model: tf.keras.Model,
        pruning_schedule,
    ) -> tf.keras.Model:
        """Clone `model` while wrapping supported layers with pruning."""

        def maybe_prune(layer: tf.keras.layers.Layer):
            if self._is_layer_prunable(layer):
                return tfmot.sparsity.keras.prune_low_magnitude(
                    layer,
                    pruning_schedule=pruning_schedule,
                )
            return layer

        # clone_model automatically copies the pretrained weights; avoid
        # calling set_weights because pruning wrappers add mask variables.
        pruned_model = tf.keras.models.clone_model(
            model,
            clone_function=maybe_prune,
        )
        return pruned_model

    def _is_layer_prunable(self, layer: tf.keras.layers.Layer) -> bool:
        registry = getattr(tfmot.sparsity.keras, "pruning_registry", None)
        if registry is not None:
            supports = getattr(registry.PruneRegistry, "supports", None)
            if callable(supports):
                try:
                    if supports(layer):
                        return True
                except Exception:
                    pass

        if isinstance(layer, PRUNABLE_LAYER_TYPES):
            return True
        get_prunable = getattr(layer, "get_prunable_weights", None)
        if callable(get_prunable):
            try:
                return bool(get_prunable())
            except Exception:
                return False
        return False

    def _safe_set_learning_rate(self, optimizer, learning_rate: Optional[float]) -> None:
        """Safely set learning rate, handling schedulers and non-settable cases."""
        if learning_rate is None:
            return
        
        # Check if optimizer uses a learning rate schedule
        if hasattr(optimizer, '_learning_rate') and hasattr(optimizer._learning_rate, '__call__'):
            # Recreate optimizer with float learning rate
            optimizer_class = optimizer.__class__
            config = optimizer.get_config()
            config['learning_rate'] = learning_rate
            
            # Handle weight decay parameter name differences
            weight_decay_key = None
            for key in ['weight_decay', 'decay']:
                if key in config:
                    weight_decay_key = key
                    break
            
            try:
                # Try to recreate with learning_rate parameter
                new_optimizer = optimizer_class(learning_rate=learning_rate, **{
                    k: v for k, v in config.items() 
                    if k not in ['learning_rate', 'name', 'weight_decay', 'decay']
                })
                # Copy weight decay if present
                if weight_decay_key and hasattr(new_optimizer, weight_decay_key):
                    setattr(new_optimizer, weight_decay_key, config.get(weight_decay_key, 0.0))
                
                # Replace the optimizer object
                for attr in dir(new_optimizer):
                    if not attr.startswith('_'):
                        try:
                            setattr(optimizer, attr, getattr(new_optimizer, attr))
                        except (AttributeError, TypeError):
                            pass
            except Exception:
                # If recreation fails, skip setting learning rate
                pass
        elif hasattr(optimizer, 'learning_rate'):
            try:
                optimizer.learning_rate = learning_rate
            except (TypeError, AttributeError):
                # Learning rate is not settable, skip
                pass

    def _clone_and_compile(
        self,
        learning_rate: Optional[float] = None,
    ) -> tf.keras.Model:
        model_clone = tf.keras.models.clone_model(self.base_model)
        model_clone.set_weights(self.base_model.get_weights())

        optimizer = tf.keras.optimizers.deserialize(self._optimizer_config)
        self._safe_set_learning_rate(optimizer, learning_rate)

        loss = tf.keras.losses.deserialize(self._loss_config, custom_objects=CUSTOM_OBJECTS)
        metrics = []
        for metric_config in self._metric_configs:
            if isinstance(metric_config, str):
                # Fallback from serialization errors - use string identifier
                metrics.append(metric_config)
            else:
                try:
                    metrics.append(tf.keras.metrics.deserialize(metric_config, custom_objects=CUSTOM_OBJECTS))
                except Exception:
                    # If deserialization fails, use categorical accuracy as safe default
                    metrics.append(tf.keras.metrics.CategoricalAccuracy(name="accuracy"))

        model_clone.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )
        return model_clone

    def _strip_and_compile(
        self,
        pruned_model: tf.keras.Model,
    ) -> tf.keras.Model:
        """Strip pruning wrappers and recompile the model."""
        stripped_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
        
        optimizer = tf.keras.optimizers.deserialize(self._optimizer_config)
        loss = tf.keras.losses.deserialize(self._loss_config, custom_objects=CUSTOM_OBJECTS)
        metrics = []
        for metric_config in self._metric_configs:
            if isinstance(metric_config, str):
                metrics.append(metric_config)
            else:
                try:
                    metrics.append(tf.keras.metrics.deserialize(metric_config, custom_objects=CUSTOM_OBJECTS))
                except Exception:
                    metrics.append(tf.keras.metrics.CategoricalAccuracy(name="accuracy"))
        
        stripped_model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )
        return stripped_model

    def _prepare_default_datasets(
        self,
        batch_size: int = 128,
    ) -> DatasetBundle:
        try:
            (
                x_train,
                y_train,
                x_val,
                y_val,
                x_test,
                y_test,
                _,
            ) = prepare_compression_datasets()
        except Exception as exc:  # pragma: no cover - dataset download dependency
            raise RuntimeError(
                "Failed to prepare CIFAR-100 data. Ensure the dataset is available "
                "locally or provide custom datasets to PruningComparator."
            ) from exc

        train_aug = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.05),
                tf.keras.layers.RandomTranslation(0.1, 0.1),
            ]
        )

        def augment(x, y):
            return train_aug(x), y

        train_ds = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(10000)
            .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices((x_val, y_val))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        test_ds = (
            tf.data.Dataset.from_tensor_slices((x_test, y_test))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        return DatasetBundle(
            train=train_ds,
            val=val_ds,
            test=test_ds,
            train_size=len(x_train),
            val_size=len(x_val),
            test_size=len(x_test),
        )

    def _resolve_dataset(
        self,
        dataset: Optional[DatasetLike],
        fallback_split: str,
        batch_size: int,
        augment: bool = False,
        shuffle: bool = False,
    ) -> tf.data.Dataset:
        if dataset is None:
            bundle = self._dataset_bundle
            if bundle is None or self._cached_batch_size != batch_size:
                bundle = self._prepare_default_datasets(batch_size=batch_size)
                if self.cache_datasets:
                    self._dataset_bundle = bundle
                    self._cached_batch_size = batch_size
            if fallback_split == "train":
                return bundle.train
            if fallback_split == "val":
                return bundle.val
            return bundle.test

        if isinstance(dataset, tf.data.Dataset):
            return dataset

        features, labels = dataset
        ds = tf.data.Dataset.from_tensor_slices((features, labels))
        if shuffle:
            ds = ds.shuffle(len(features))
        if augment:
            augmentor = tf.keras.Sequential(
                [
                    tf.keras.layers.RandomFlip("horizontal"),
                    tf.keras.layers.RandomRotation(0.05),
                ]
            )

            def apply_aug(x, y):
                return augmentor(x), y

            ds = ds.map(apply_aug, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def _resolve_dataset_size(
        self,
        dataset: Optional[DatasetLike],
        default: str,
    ) -> Optional[int]:
        if dataset is None:
            if self._dataset_bundle is None:
                return None
            if default == "train":
                return self._dataset_bundle.train_size
            if default == "val":
                return self._dataset_bundle.val_size
            return self._dataset_bundle.test_size

        if isinstance(dataset, tf.data.Dataset):
            cardinality = tf.data.experimental.cardinality(dataset).numpy()
            if cardinality == tf.data.experimental.UNKNOWN_CARDINALITY:
                return None
            return cardinality

        return len(dataset[0])

    def _build_pruning_schedule(
        self,
        schedule_name: str,
        target_sparsity: float,
        fine_tune_epochs: int,
        batch_size: int,
        train_size: Optional[int],
    ):
        steps_per_epoch = (
            math.ceil(train_size / batch_size)
            if train_size is not None
            else 100
        )
        begin_step = 0
        end_step = steps_per_epoch * fine_tune_epochs

        schedule_name = schedule_name.lower()
        if schedule_name == "polynomial":
            return tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=target_sparsity,
                begin_step=begin_step,
                end_step=end_step,
            )
        if schedule_name == "gradual":
            return tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=target_sparsity,
                begin_step=begin_step,
                end_step=end_step,
                power=0.5,
            )
        if schedule_name == "constant":
            return tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=target_sparsity,
                begin_step=begin_step,
                frequency=max(1, steps_per_epoch // 2),
            )
        raise ValueError(f"Unsupported pruning schedule: {schedule_name}")

    def _compute_layer_sparsity(
        self,
        model: tf.keras.Model,
    ) -> Dict[str, Dict[str, float]]:
        analysis: Dict[str, Dict[str, float]] = {}
        for layer in model.layers:
            weights = layer.get_weights()
            if not weights:
                continue
            total_params = sum(weight.size for weight in weights)
            zero_params = sum(np.sum(np.isclose(weight, 0.0)) for weight in weights)
            analysis[layer.name] = {
                "sparsity": zero_params / total_params if total_params else 0.0,
                "parameters": total_params,
            }
        return analysis

    def _evaluate_accuracy(
        self,
        model: tf.keras.Model,
        dataset: tf.data.Dataset,
    ) -> float:
        metrics = model.evaluate(dataset, verbose=0)
        if isinstance(metrics, list):
            if len(metrics) == 1:
                return float(metrics[0])
            return float(metrics[1])
        return float(metrics)

    def _compute_gradient_importance(
        self,
        model: tf.keras.Model,
        layer_name: str,
        dataset: tf.data.Dataset,
        num_batches: int = 10,
    ) -> np.ndarray:
        """Compute gradient-based importance for filters."""
        layer = model.get_layer(layer_name)
        if not isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            raise ValueError(f"Layer {layer_name} is not a Conv2D or Dense layer")
        
        if isinstance(layer, tf.keras.layers.Conv2D):
            kernel_shape = layer.kernel.shape
            filter_gradients = np.zeros(kernel_shape[-1])
        else:
            kernel_shape = layer.kernel.shape
            filter_gradients = np.zeros(kernel_shape[-1])
        
        batch_count = 0
        for images, labels in dataset.take(num_batches):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = model.loss(labels, predictions)
            
            gradients = tape.gradient(loss, layer.kernel)
            if gradients is not None:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    filter_gradients += np.sum(np.abs(gradients.numpy()), axis=(0, 1, 2))
                else:
                    filter_gradients += np.sum(np.abs(gradients.numpy()), axis=0)
            
            batch_count += 1
            if batch_count >= num_batches:
                break
        
        if batch_count > 0:
            filter_gradients /= batch_count
        
        return filter_gradients
    
    def _compute_taylor_importance(
        self,
        model: tf.keras.Model,
        layer_name: str,
        dataset: tf.data.Dataset,
        num_batches: int = 10,
    ) -> np.ndarray:
        """Compute Taylor importance (gradient * weight) for filters."""
        layer = model.get_layer(layer_name)
        if not isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            raise ValueError(f"Layer {layer_name} is not a Conv2D or Dense layer")
        
        kernel = layer.kernel.numpy()
        
        if isinstance(layer, tf.keras.layers.Conv2D):
            filter_importance = np.zeros(kernel.shape[-1])
        else:
            filter_importance = np.zeros(kernel.shape[-1])
        
        batch_count = 0
        for images, labels in dataset.take(num_batches):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = model.loss(labels, predictions)
            
            gradients = tape.gradient(loss, layer.kernel)
            if gradients is not None:
                grad_np = gradients.numpy()
                if isinstance(layer, tf.keras.layers.Conv2D):
                    taylor = np.abs(kernel * grad_np)
                    filter_importance += np.sum(taylor, axis=(0, 1, 2))
                else:
                    taylor = np.abs(kernel * grad_np)
                    filter_importance += np.sum(taylor, axis=0)
            
            batch_count += 1
            if batch_count >= num_batches:
                break
        
        if batch_count > 0:
            filter_importance /= batch_count
        
        return filter_importance

    def _compute_filter_importance(
        self,
        kernel: np.ndarray,
        metric: str,
        model: Optional[tf.keras.Model] = None,
        layer_name: Optional[str] = None,
        dataset: Optional[tf.data.Dataset] = None,
    ) -> np.ndarray:
        """Compute filter importance using various metrics."""
        metric = metric.lower()
        
        if metric == "l2_norm":
            return np.sqrt(np.sum(np.square(kernel), axis=(0, 1, 2)))
        
        elif metric == "gradient":
            if model is None or layer_name is None or dataset is None:
                raise ValueError("Gradient importance requires model, layer_name, and dataset")
            return self._compute_gradient_importance(model, layer_name, dataset)
        
        elif metric == "taylor":
            if model is None or layer_name is None or dataset is None:
                raise ValueError("Taylor importance requires model, layer_name, and dataset")
            return self._compute_taylor_importance(model, layer_name, dataset)
        
        else:  # default l1
            return np.sum(np.abs(kernel), axis=(0, 1, 2))

    def _find_previous_conv(
        self,
        layer: tf.keras.layers.Layer,
        model: tf.keras.Model,
    ) -> Optional[str]:
        layer_index = None
        for idx, current_layer in enumerate(model.layers):
            if current_layer.name == layer.name:
                layer_index = idx
                break
        if layer_index is None:
            return None
        for idx in range(layer_index - 1, -1, -1):
            if isinstance(model.layers[idx], tf.keras.layers.Conv2D):
                return model.layers[idx].name
        return None


class LayerDependencyAnalyzer:
    """Analyzes layer dependencies for structured pruning."""
    
    def __init__(self, model: tf.keras.Model):
        self.model = model
        self.dependency_graph = self._build_dependency_graph()
    
    def _build_dependency_graph(self) -> Dict[str, Dict]:
        """Build a graph of layer dependencies."""
        graph = {}
        
        layer_map = {layer.name: layer for layer in self.model.layers}
        
        for layer in self.model.layers:
            outbound_layers = self._find_outbound_layers(layer.name, layer_map)
            inbound_layers = self._find_inbound_layers(layer.name, layer_map)
            
            graph[layer.name] = {
                'layer': layer,
                'inbound': inbound_layers,
                'outbound': outbound_layers,
                'output_shape': layer.output_shape if hasattr(layer, 'output_shape') else None,
                'input_shape': layer.input_shape if hasattr(layer, 'input_shape') else None
            }
        
        return graph
    
    def _find_outbound_layers(self, layer_name: str, layer_map: Dict) -> List[str]:
        """Find layers that receive output from the given layer."""
        outbound = []
        for layer in self.model.layers:
            if hasattr(layer, 'input'):
                try:
                    inbound_layers = layer.input
                    if isinstance(inbound_layers, list):
                        for inp in inbound_layers:
                            if hasattr(inp, '_keras_history'):
                                src_layer = inp._keras_history[0]
                                if src_layer.name == layer_name:
                                    outbound.append(layer.name)
                    else:
                        if hasattr(inbound_layers, '_keras_history'):
                            src_layer = inbound_layers._keras_history[0]
                            if src_layer.name == layer_name:
                                outbound.append(layer.name)
                except:
                    pass
            else:
                layer_idx = list(layer_map.keys()).index(layer_name)
                current_idx = list(layer_map.keys()).index(layer.name)
                if current_idx == layer_idx + 1:
                    outbound.append(layer.name)
        
        return list(set(outbound))
    
    def _find_inbound_layers(self, layer_name: str, layer_map: Dict) -> List[str]:
        """Find layers that provide input to the given layer."""
        inbound = []
        layer = layer_map[layer_name]
        
        if hasattr(layer, 'input'):
            try:
                inbound_inputs = layer.input
                if isinstance(inbound_inputs, list):
                    for inp in inbound_inputs:
                        if hasattr(inp, '_keras_history'):
                            src_layer = inp._keras_history[0]
                            inbound.append(src_layer.name)
                else:
                    if hasattr(inbound_inputs, '_keras_history'):
                        src_layer = inbound_inputs._keras_history[0]
                        inbound.append(src_layer.name)
            except:
                pass
        
        if not inbound:
            layer_names = list(layer_map.keys())
            try:
                idx = layer_names.index(layer_name)
                if idx > 0:
                    inbound.append(layer_names[idx - 1])
            except:
                pass
        
        return list(set(inbound))
    
    def propagate_pruning_effects(
        self,
        pruned_layers: Dict[str, List[int]]
    ) -> Dict[str, Tuple[int, int]]:
        """Propagate the effects of pruning through the network."""
        dimension_changes = {}
        
        for layer_name, info in self.dependency_graph.items():
            layer = info['layer']
            if hasattr(layer, 'filters'):
                dimension_changes[layer_name] = (layer.filters, layer.filters)
            elif hasattr(layer, 'units'):
                dimension_changes[layer_name] = (layer.units, layer.units)
            else:
                dimension_changes[layer_name] = (None, None)
        
        for layer_name, removed_filters in pruned_layers.items():
            if layer_name not in self.dependency_graph:
                continue
            
            layer_info = self.dependency_graph[layer_name]
            layer = layer_info['layer']
            
            if hasattr(layer, 'filters'):
                original_filters = layer.filters
                new_output_channels = original_filters - len(removed_filters)
                
                current_input, _ = dimension_changes[layer_name]
                dimension_changes[layer_name] = (current_input, new_output_channels)
                
                queue = [(outbound, new_output_channels) for outbound in layer_info['outbound']]
                
                while queue:
                    current_layer_name, new_input_channels = queue.pop(0)
                    if current_layer_name not in dimension_changes:
                        continue
                    
                    current_input, current_output = dimension_changes[current_layer_name]
                    
                    dimension_changes[current_layer_name] = (new_input_channels, current_output)
                    
                    current_layer_info = self.dependency_graph.get(current_layer_name)
                    if current_layer_info:
                        current_layer = current_layer_info['layer']
                        if hasattr(current_layer, 'units'):
                            new_output = current_layer.units
                            dimension_changes[current_layer_name] = (new_input_channels, new_output)
                            for next_layer in current_layer_info['outbound']:
                                queue.append((next_layer, new_output))
        
        return dimension_changes
    
    def get_affected_layers(self, pruned_layer_name: str) -> List[str]:
        """Get all layers affected by pruning a specific layer."""
        affected = set()
        queue = [pruned_layer_name]
        
        while queue:
            current = queue.pop(0)
            if current in affected:
                continue
            
            affected.add(current)
            
            if current in self.dependency_graph:
                for outbound in self.dependency_graph[current]['outbound']:
                    queue.append(outbound)
        
        affected.discard(pruned_layer_name)
        return list(affected)


class ModelRebuilder:
    """Rebuilds model with pruned layers."""
    
    def __init__(self, original_model: tf.keras.Model):
        self.original_model = original_model
        self.original_weights = {layer.name: layer.get_weights() for layer in original_model.layers}
    
    def rebuild_with_pruned_filters(
        self,
        filters_to_remove: Dict[str, List[int]],
        dependency_analyzer: LayerDependencyAnalyzer
    ) -> tf.keras.Model:
        """Rebuild model with physically removed filters."""
        dimension_changes = dependency_analyzer.propagate_pruning_effects(filters_to_remove)
        
        new_layers = []
        layer_configs = {}
        
        for layer in self.original_model.layers:
            layer_name = layer.name
            
            if layer_name in filters_to_remove:
                new_layer = self._create_pruned_layer(layer, filters_to_remove[layer_name])
            else:
                new_input_dim, new_output_dim = dimension_changes.get(layer_name, (None, None))
                
                if new_input_dim is not None and hasattr(layer, 'units'):
                    new_layer = self._create_dense_with_new_input(layer, new_input_dim)
                elif new_input_dim is not None and hasattr(layer, 'filters'):
                    new_layer = self._create_conv_with_new_input(layer, new_input_dim)
                else:
                    new_layer = layer.__class__.from_config(layer.get_config())
            
            new_layers.append(new_layer)
            layer_configs[layer_name] = new_layer
        
        return self._rebuild_model_topology(new_layers, layer_configs, dimension_changes, filters_to_remove)
    
    def _create_pruned_layer(self, layer: tf.keras.layers.Layer, filters_to_remove: List[int]) -> tf.keras.layers.Layer:
        """Create a new layer with pruned filters/neurons."""
        config = layer.get_config().copy()
        
        if isinstance(layer, tf.keras.layers.Conv2D):
            original_filters = config['filters']
            new_filters = original_filters - len(filters_to_remove)
            config['filters'] = max(1, new_filters)
            
        elif isinstance(layer, tf.keras.layers.Dense):
            original_units = config['units']
            new_units = original_units - len(filters_to_remove)
            config['units'] = max(1, new_units)
            
        return layer.__class__.from_config(config)
    
    def _create_dense_with_new_input(self, layer: tf.keras.layers.Dense, new_input_dim: int) -> tf.keras.layers.Dense:
        """Create Dense layer with new input dimension."""
        config = layer.get_config().copy()
        return tf.keras.layers.Dense.from_config(config)
    
    def _create_conv_with_new_input(self, layer: tf.keras.layers.Conv2D, new_input_dim: int) -> tf.keras.layers.Conv2D:
        """Create Conv2D layer with new input dimension."""
        config = layer.get_config().copy()
        return tf.keras.layers.Conv2D.from_config(config)
    
    def _rebuild_model_topology(
        self,
        new_layers: List[tf.keras.layers.Layer],
        layer_configs: Dict[str, tf.keras.layers.Layer],
        dimension_changes: Dict[str, Tuple[int, int]],
        filters_to_remove: Dict[str, List[int]]
    ) -> tf.keras.Model:
        """Rebuild model topology with new layers."""
        if isinstance(self.original_model, tf.keras.Sequential):
            new_model = tf.keras.Sequential()
            for layer in new_layers:
                new_model.add(layer)
        else:
            new_model = self._rebuild_functional_model(new_layers, dimension_changes)
        
        self._copy_weights_to_pruned_model(new_model, filters_to_remove, dimension_changes)
        
        return new_model
    
    def _rebuild_functional_model(
        self,
        new_layers: List[tf.keras.layers.Layer],
        dimension_changes: Dict[str, Tuple[int, int]]
    ) -> tf.keras.Model:
        """Rebuild functional API model."""
        first_layer = new_layers[0]
        if hasattr(first_layer, 'input_shape') and first_layer.input_shape:
            input_shape = first_layer.input_shape[1:]
            inputs = tf.keras.Input(shape=input_shape)
        else:
            inputs = tf.keras.Input(shape=(32, 32, 3))
        
        x = inputs
        for layer in new_layers:
            x = layer(x)
        
        return tf.keras.Model(inputs=inputs, outputs=x)
    
    def _copy_weights_to_pruned_model(
        self,
        new_model: tf.keras.Model,
        filters_to_remove: Dict[str, List[int]],
        dimension_changes: Dict[str, Tuple[int, int]]
    ):
        """Copy weights from original model to pruned model."""
        for new_layer in new_model.layers:
            layer_name = new_layer.name
            
            if layer_name not in self.original_weights:
                continue
            
            original_weights = self.original_weights[layer_name]
            
            if not original_weights:
                continue
            
            if layer_name in filters_to_remove:
                pruned_weights = self._extract_kept_weights(
                    original_weights, filters_to_remove[layer_name], layer_name
                )
                new_layer.set_weights(pruned_weights)
            else:
                new_input_dim, _ = dimension_changes.get(layer_name, (None, None))
                
                if new_input_dim is not None and len(original_weights) > 0:
                    adjusted_weights = self._adjust_weights_for_new_input(
                        original_weights, new_input_dim, layer_name
                    )
                    new_layer.set_weights(adjusted_weights)
                else:
                    new_layer.set_weights(original_weights)
    
    def _extract_kept_weights(
        self,
        weights: List[np.ndarray],
        removed_indices: List[int],
        layer_name: str
    ) -> List[np.ndarray]:
        """Extract weights for kept filters/neurons."""
        if not weights:
            return weights
        
        kernel = weights[0]
        bias = weights[1] if len(weights) > 1 else None
        
        num_filters = kernel.shape[-1]
        keep_mask = np.ones(num_filters, dtype=bool)
        keep_mask[removed_indices] = False
        
        new_kernel = kernel[..., keep_mask]
        
        new_weights = [new_kernel]
        
        if bias is not None:
            new_bias = bias[keep_mask]
            new_weights.append(new_bias)
        
        for i in range(2, len(weights)):
            new_weights.append(weights[i][keep_mask] if weights[i].shape[0] == num_filters else weights[i])
        
        return new_weights
    
    def _adjust_weights_for_new_input(
        self,
        weights: List[np.ndarray],
        new_input_dim: int,
        layer_name: str
    ) -> List[np.ndarray]:
        """Adjust weights for changed input dimension."""
        if not weights:
            return weights
        
        kernel = weights[0]
        original_input_dim = kernel.shape[-2] if len(kernel.shape) > 1 else kernel.shape[-1]
        
        if original_input_dim == new_input_dim:
            return weights
        
        if len(kernel.shape) == 4:
            new_kernel = kernel[..., :new_input_dim, :]
        else:
            new_kernel = kernel[..., :new_input_dim, :]
        
        new_weights = [new_kernel]
        
        for i in range(1, len(weights)):
            new_weights.append(weights[i])
        
        return new_weights
