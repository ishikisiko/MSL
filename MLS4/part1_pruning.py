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
    ) -> Dict:
        """Perform magnitude-based pruning with configurable schedules."""

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
        return results

    def structured_pruning(
        self,
        target_reduction: float = 0.5,
        importance_metric: str = "l1_norm",
        fine_tune_epochs: int = 4,
        learning_rate: float = 5e-4,
        batch_size: int = 128,
        train_data: Optional[DatasetLike] = None,
        val_data: Optional[DatasetLike] = None,
    ) -> Dict:
        """Apply coarse-grained filter/channel pruning."""

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
        return results

    def gradual_vs_oneshot_pruning(
        self,
        target_sparsity: float = 0.7,
        fine_tune_epochs: int = 6,
        batch_size: int = 128,
    ) -> Dict:
        """Compare gradual polynomial pruning vs one-shot constant pruning."""

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
        return comparison

    def analyze_pruning_sensitivity(
        self,
        probe_sparsity: float = 0.3,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        """Individually prune each layer to measure accuracy drop."""

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
        """Wrap supported layers with pruning masks without reassigning weights."""

        pruning_params = {
            "pruning_schedule": pruning_schedule,
        }

        # tfmot handles cloning and initializes pruning metadata while
        # preserving the original layer weights. Avoid manual set_weights
        # because pruned layers introduce extra mask variables.
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
            model,
            **pruning_params,
        )
        return pruned_model

    def _is_layer_prunable(self, layer: tf.keras.layers.Layer) -> bool:
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

    def _compute_filter_importance(
        self,
        kernel: np.ndarray,
        metric: str,
    ) -> np.ndarray:
        metric = metric.lower()
        if metric == "l2_norm":
            importance = np.sqrt(np.sum(np.square(kernel), axis=(0, 1, 2)))
        elif metric == "gradient":
            importance = np.sum(np.abs(kernel), axis=(0, 1, 2))
        elif metric == "taylor":
            importance = np.sum(np.square(kernel), axis=(0, 1, 2))
        else:  # default l1
            importance = np.sum(np.abs(kernel), axis=(0, 1, 2))
        return importance

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
