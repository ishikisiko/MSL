from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tf_compat  # noqa: F401  # align TensorFlow with tfmot expectations
import tensorflow as tf
from tensorflow.keras import mixed_precision

# CRITICAL: Force disable XLA JIT compilation to avoid EfficientNet layout errors
try:
    tf.config.optimizer.set_jit(False)
    # Disable grappler meta optimizer which can invoke XLA
    tf.config.optimizer.set_experimental_options({
        'disable_meta_optimizer': True,
        'disable_model_pruning': True,
        'constant_folding': False,
        'arithmetic_optimization': False,
        'layout_optimizer': False,
        'dependency_optimization': False,
        'shape_optimization': False,
        'remapping': False,
        'scoped_allocator_optimization': False,
        'implementation_selector': False,
        'auto_mixed_precision': False,
    })
    print("✓ XLA JIT and Grappler optimizer disabled successfully")
except Exception as e:
    print(f"⚠ Warning: Could not fully disable optimizers: {e}")

from baseline_model import prepare_compression_datasets


AUTOTUNE = tf.data.AUTOTUNE


class _precision_guard:
    """Context manager that temporarily switches mixed precision policy."""

    def __init__(self, policy_name: str) -> None:
        self._target = policy_name
        self._original = mixed_precision.global_policy()

    def __enter__(self) -> None:
        if mixed_precision.global_policy().name != self._target:
            mixed_precision.set_global_policy(self._target)

    def __exit__(self, exc_type, exc, tb) -> None:
        if mixed_precision.global_policy().name != self._original.name:
            mixed_precision.set_global_policy(self._original)


@dataclass
class DatasetBundle:
    train: tf.data.Dataset
    val: tf.data.Dataset
    test: tf.data.Dataset
    train_size: int
    val_size: int
    test_size: int


class SimpleDistiller(tf.keras.Model):
    """Lightweight distillation wrapper from the official TensorFlow tutorial."""

    def __init__(
        self,
        student: tf.keras.Model,
        teacher: tf.keras.Model,
        temperature: float = 5.0,
        alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
        self.student_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self.distillation_loss_fn = tf.keras.losses.KLDivergence()

    def compile(self, optimizer, metrics, **kwargs):
        # Disable XLA JIT compilation to avoid layout errors with EfficientNet Dropout layers
        kwargs['jit_compile'] = False
        super().compile(optimizer=optimizer, metrics=metrics, **kwargs)

    def train_step(self, data):
        x, y = data
        x = tf.cast(x, tf.float32)
        # Stop gradient to avoid backprop through teacher's BatchNorm layers
        teacher_predictions = tf.stop_gradient(self.teacher(x, training=False))
        temperature = self.temperature
        alpha = self.alpha

        with tf.GradientTape() as tape:
            student_predictions = self.student(x, training=True)
            student_loss = self.student_loss_fn(y, student_predictions)
            teacher_soft = tf.nn.softmax(teacher_predictions / temperature, axis=-1)
            student_soft = tf.nn.softmax(student_predictions / temperature, axis=-1)
            distillation_loss = self.distillation_loss_fn(
                teacher_soft, student_soft
            ) * (temperature**2)
            loss = alpha * student_loss + (1 - alpha) * distillation_loss

        gradients = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
        self.compiled_metrics.update_state(y, student_predictions)
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {
                "student_loss": student_loss,
                "distillation_loss": distillation_loss,
            }
        )
        return results

    def test_step(self, data):
        x, y = data
        x = tf.cast(x, tf.float32)
        predictions = self.student(x, training=False)
        student_loss = self.student_loss_fn(y, predictions)
        self.compiled_metrics.update_state(y, predictions)
        results = {m.name: m.result() for m in self.metrics}
        results["student_loss"] = student_loss
        return results


class DistillationFramework:
    """
    Implements the experiments for Part III (knowledge distillation).

    Args:
        teacher_model: Pre-trained teacher model.
        student_architecture: Callable that returns a freshly compiled student model.
            The callable receives a single float argument representing a width/size
            multiplier, allowing progressive/self distillation to request smaller
            or intermediate backbones.
    """

    def __init__(
        self,
        teacher_model: tf.keras.Model,
        student_architecture: Callable[[float], tf.keras.Model],
        cache_datasets: bool = True,
        batch_size: int = 32,
    ) -> None:
        # No need to switch precision - baseline is already FP32
        self._original_policy = mixed_precision.global_policy()
        
        self.teacher = self._clone_model_float32(teacher_model)
        self.student_arch = student_architecture
        self.distillation_results: Dict[str, Dict] = {}
        self.cache_datasets = cache_datasets
        self.batch_size = int(batch_size)
        self._dataset_bundle: Optional[DatasetBundle] = None
        if cache_datasets:
            self._dataset_bundle = self._prepare_datasets()

    def _clone_model_float32(self, model: tf.keras.Model) -> tf.keras.Model:
        """Clone a model under float32 policy to avoid dtype mismatches."""

        with _precision_guard("float32"):
            clone = tf.keras.models.clone_model(model)
        weights = [np.asarray(weight).astype(np.float32) for weight in model.get_weights()]
        clone.set_weights(weights)
        if getattr(model, "optimizer", None):
            try:
                optimizer = tf.keras.optimizers.deserialize(
                    tf.keras.optimizers.serialize(model.optimizer)
                )
            except Exception:
                optimizer = tf.keras.optimizers.Adam()
        else:
            optimizer = tf.keras.optimizers.Adam()

        loss = getattr(model, "loss", None)
        if loss is None:
            loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = getattr(model, "metrics", None) or [
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        ]
        clone.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return clone

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def temperature_optimization(
        self,
        temperature_range: Tuple[float, float] = (1.0, 20.0),
        num_trials: int = 5,
        width_multiplier: float = 0.5,
        epochs: int = 8,
        steps_per_epoch: int = 50,
    ) -> Dict:
        """
        Grid-search temperature for standard response-based distillation.
        """

        # Remove .take().repeat() to fix OUT_OF_RANGE errors
        train_ds = self._get_dataset("train")
        val_ds = self._get_dataset("val")
        temps = np.linspace(temperature_range[0], temperature_range[1], num_trials)

        trials = []
        best_model = None
        best_acc = -np.inf
        for temperature in temps:
            student = self._build_student(width_multiplier)
            distiller = SimpleDistiller(
                student=student, teacher=self.teacher, temperature=float(temperature)
            )
            distiller.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            )
            # The train dataset is already repeated in _prepare_datasets;
            # using the dataset directly and passing steps_per_epoch to
            # .fit avoids exhausting a finite iterator across epochs.
            epoch_train_ds = train_ds
            
            print(f"开始训练: temperature={temperature:.2f}, epochs={epochs}, steps_per_epoch={steps_per_epoch}")
            
            history = distiller.fit(
                epoch_train_ds,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_ds.take(max(1, steps_per_epoch // 4)),
                verbose=2,  # Show progress bars
            )
            accuracy = float(
                student.evaluate(val_ds, verbose=0)[1]
            )  # type: ignore[index]
            trial_result = {
                "temperature": float(temperature),
                "accuracy": accuracy,
                "history": history.history,
                "student_model": student,
            }
            trials.append(trial_result)
            if accuracy > best_acc:
                best_acc = accuracy
                best_model = student

        results = {
            "optimal_temperature": max(trials, key=lambda t: t["accuracy"])["temperature"],
            "temperature_accuracy_curve": {trial["temperature"]: trial["accuracy"] for trial in trials},
            "soft_target_analysis": self._analyze_soft_targets(temps),
            "knowledge_transfer_metrics": trials,
            "best_model": best_model,
        }
        self.distillation_results["temperature_search"] = results
        return results

    def progressive_distillation(
        self,
        intermediate_sizes: Sequence[float] = (0.75, 0.5, 0.25),
        temperature: float = 5.0,
        epochs: int = 8,
        steps_per_epoch: int = 40,
    ) -> Dict:
        """
        Chain teacher -> intermediate -> student distillation.
        """

        # Remove .take().repeat() to fix OUT_OF_RANGE errors
        train_ds = self._get_dataset("train")
        val_ds = self._get_dataset("val")

        current_teacher = self.teacher
        stage_results = []
        for ratio in intermediate_sizes:
            student = self._build_student(ratio)
            distiller = SimpleDistiller(
                student=student, teacher=current_teacher, temperature=temperature
            )
            distiller.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            )
            # The train dataset is already repeated in _prepare_datasets;
            # pass it directly so Keras pulls steps_per_epoch batches each
            # epoch from the repeating dataset.
            epoch_train_ds = train_ds
            history = distiller.fit(
                epoch_train_ds,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_ds.take(max(1, steps_per_epoch // 4)),
                verbose=0,
            )
            accuracy = float(student.evaluate(val_ds, verbose=0)[1])  # type: ignore[index]
            stage_results.append(
                {
                    "teacher": current_teacher.name,
                    "student_ratio": float(ratio),
                    "accuracy": accuracy,
                    "history": history.history,
                    "student_model": student,
                }
            )
            current_teacher = student

        if not stage_results:
            results = {
                "intermediate_models": [],
                "progressive_chain_results": [],
                "direct_distillation_comparison": 0.0,
                "knowledge_preservation_analysis": [],
                "final_student": None,
            }
            self.distillation_results["progressive"] = results
            return results

        results = {
            "intermediate_models": stage_results,
            "progressive_chain_results": stage_results,
            "direct_distillation_comparison": (
                stage_results[-1]["accuracy"]
                - self.temperature_optimization(
                    temperature_range=(temperature, temperature),
                    num_trials=1,
                    width_multiplier=intermediate_sizes[-1],
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch,
                )["knowledge_transfer_metrics"][0]["accuracy"]
            ),
            "knowledge_preservation_analysis": [
                {"stage": idx + 1, "accuracy": stage["accuracy"]}
                for idx, stage in enumerate(stage_results)
            ],
            "final_student": stage_results[-1]["student_model"],
        }
        self.distillation_results["progressive"] = results
        return results

    def attention_transfer(
        self,
        layer_names: Optional[Sequence[str]] = None,
        weight: float = 1.0,
        width_multiplier: float = 0.5,
        epochs: int = 15,
        steps_per_epoch: int = 40,
    ) -> Dict:
        """
        Transfer spatial attention maps alongside logits.
        """

        student = self._build_student(width_multiplier)
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        val_ds = self._get_dataset("val")
        # Remove .take().repeat() to fix OUT_OF_RANGE errors
        train_ds = self._get_dataset("train")

        teacher_layers = list(layer_names or self._default_attention_layers(self.teacher))
        student_layer_names = {layer.name for layer in student.layers}
        student_layers = [name for name in teacher_layers if name in student_layer_names]
        if not student_layers:
            fallback = [
                layer.name
                for layer in student.layers
                if isinstance(layer, tf.keras.layers.Conv2D)
            ]
            student_layers = fallback[-3:] if fallback else []
            teacher_layers = student_layers

        if not student_layers:
            raise RuntimeError(
                "Unable to find overlapping convolution layers for attention transfer."
            )
        teacher_attention_model = tf.keras.Model(
            self.teacher.inputs,
            [self.teacher.get_layer(name).output for name in teacher_layers],
        )
        student_attention_model = tf.keras.Model(
            student.inputs,
            [student.get_layer(name).output for name in student_layers],
        )

        history = {"loss": [], "attention_loss": [], "accuracy": []}
        metric = tf.keras.metrics.SparseCategoricalAccuracy()

        for _ in range(epochs):
            epoch_loss = []
            epoch_attention_loss = []
            metric.reset_state()
            for images, labels in train_ds.take(steps_per_epoch):
                with tf.GradientTape() as tape:
                    student_logits = student(images, training=True)
                    # Stop gradient to avoid backprop through teacher's BatchNorm layers
                    teacher_att = teacher_attention_model(images, training=False)
                    teacher_att = [tf.stop_gradient(feat) for feat in teacher_att]
                    student_att = student_attention_model(images, training=True)
                    attention_loss = self._attention_loss(teacher_att, student_att)
                    ce_loss = tf.reduce_mean(
                        tf.keras.losses.sparse_categorical_crossentropy(
                            labels, student_logits
                        )
                    )
                    loss = ce_loss + weight * attention_loss
                grads = tape.gradient(loss, student.trainable_variables)
                optimizer.apply_gradients(zip(grads, student.trainable_variables))
                metric.update_state(labels, student_logits)
                epoch_loss.append(float(loss))
                epoch_attention_loss.append(float(attention_loss))
            history["loss"].append(float(np.mean(epoch_loss)))
            history["attention_loss"].append(float(np.mean(epoch_attention_loss)))
            history["accuracy"].append(float(metric.result()))

        accuracy = float(student.evaluate(val_ds, verbose=0)[1])  # type: ignore[index]
        results = {
            "attention_maps_analysis": {"layers": list(teacher_layers)},
            "attention_transfer_loss": history["attention_loss"],
            "combined_distillation_results": {
                "model": student,
                "accuracy": accuracy,
                "history": history,
            },
            "spatial_pattern_analysis": {
                "mean_attention_loss": float(np.mean(history["attention_loss"])),
                "final_accuracy": accuracy,
            },
        }
        self.distillation_results["attention_transfer"] = results
        return results

    def feature_matching_distillation(
        self,
        layer_pairs: Optional[Sequence[Tuple[str, str]]] = None,
        width_multiplier: float = 0.5,
        epochs: int = 10,
        steps_per_epoch: int = 30,
    ) -> Dict:
        """
        Align intermediate representations between teacher and student.
        """
        
        with _precision_guard("float32"):
            student = self._build_student(width_multiplier)
            optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
            metric = tf.keras.metrics.SparseCategoricalAccuracy()
            val_ds = self._get_dataset("val")

            if layer_pairs is None:
                teacher_convs = [layer.name for layer in self.teacher.layers if isinstance(layer, tf.keras.layers.Conv2D)]
                student_convs = [layer.name for layer in student.layers if isinstance(layer, tf.keras.layers.Conv2D)]
                layer_pairs = list(zip(teacher_convs[-3:], student_convs[-3:]))
            layer_pairs = [pair for pair in layer_pairs if pair[0] and pair[1]]
            if not layer_pairs:
                raise RuntimeError("No convolutional layer pairs available for feature matching.")

            teacher_features = tf.keras.Model(
                self.teacher.inputs,
                [self.teacher.get_layer(t_name).output for t_name, _ in layer_pairs],
            )
            student_features = tf.keras.Model(
                student.inputs,
                [student.get_layer(s_name).output for _, s_name in layer_pairs],
            )

            history = {"feature_loss": [], "accuracy": []}
            # Remove .take().repeat() to fix OUT_OF_RANGE errors
            train_ds = self._get_dataset("train")

            for _ in range(epochs):
                metric.reset_state()
                feature_losses = []
                for images, labels in train_ds.take(steps_per_epoch):
                    with tf.GradientTape() as tape:
                        student_logits = student(images, training=True)
                        # Stop gradient to avoid backprop through teacher's BatchNorm layers
                        teacher_acts = teacher_features(images, training=False)
                        teacher_acts = [tf.stop_gradient(feat) for feat in teacher_acts]
                        student_acts = student_features(images, training=True)
                        feat_loss = self._feature_loss(teacher_acts, student_acts)
                        ce_loss = tf.reduce_mean(
                            tf.keras.losses.sparse_categorical_crossentropy(
                                labels, student_logits
                            )
                        )
                        loss = ce_loss + 0.3 * feat_loss
                    grads = tape.gradient(loss, student.trainable_variables)
                    optimizer.apply_gradients(zip(grads, student.trainable_variables))
                    metric.update_state(labels, student_logits)
                    feature_losses.append(float(feat_loss))
                history["feature_loss"].append(float(np.mean(feature_losses)))
                history["accuracy"].append(float(metric.result()))

            accuracy = float(student.evaluate(val_ds, verbose=0)[1])  # type: ignore[index]
            results = {
                "layer_pairs": layer_pairs,
                "feature_loss_history": history["feature_loss"],
                "student_model": student,
                "accuracy": accuracy,
            }
        self.distillation_results["feature_matching"] = results
        return results

    def self_distillation_experiments(
        self,
        width_multiplier: float = 0.5,
        ensemble_size: int = 2,
        epochs: int = 10,
        steps_per_epoch: int = 30,
    ) -> Dict:
        """
        Distill from an ensemble of noisy teachers built from the baseline.
        """
        
        with _precision_guard("float32"):
            teachers = [self._noisy_teacher_copy(scale=0.01 * (idx + 1)) for idx in range(ensemble_size)]
            student = self._build_student(width_multiplier)
            optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
            metric = tf.keras.metrics.SparseCategoricalAccuracy()
            val_ds = self._get_dataset("val")
            # Remove .take().repeat() to fix OUT_OF_RANGE errors
            train_ds = self._get_dataset("train")

            history = {"loss": [], "accuracy": []}
            for _ in range(epochs):
                metric.reset_state()
                losses = []
                for images, labels in train_ds.take(steps_per_epoch):
                    with tf.GradientTape() as tape:
                        student_logits = student(images, training=True)
                        # Stop gradient to avoid backprop through teachers' BatchNorm layers
                        ensemble_outputs = [teacher(images, training=False) for teacher in teachers]
                        ensemble_outputs = [tf.stop_gradient(output) for output in ensemble_outputs]
                        ensemble_logits = tf.reduce_mean(
                            tf.stack(ensemble_outputs, axis=0),
                            axis=0,
                        )
                        hard_loss = tf.reduce_mean(
                            tf.keras.losses.sparse_categorical_crossentropy(labels, student_logits)
                        )
                        soft_loss = tf.reduce_mean(
                            tf.keras.losses.kullback_leibler_divergence(
                                tf.nn.softmax(ensemble_logits, axis=-1),
                                tf.nn.softmax(student_logits, axis=-1),
                            )
                        )
                        loss = 0.5 * hard_loss + 0.5 * soft_loss
                    grads = tape.gradient(loss, student.trainable_variables)
                    optimizer.apply_gradients(zip(grads, student.trainable_variables))
                    metric.update_state(labels, student_logits)
                    losses.append(float(loss))
                history["loss"].append(float(np.mean(losses)))
                history["accuracy"].append(float(metric.result()))

            accuracy = float(student.evaluate(val_ds, verbose=0)[1])  # type: ignore[index]
            results = {
                "ensemble_teacher_accuracy": [
                    float(teacher.evaluate(val_ds, verbose=0)[1]) for teacher in teachers
                ],
                "student_model": student,
                "accuracy": accuracy,
                "history": history,
            }
        self.distillation_results["self_distillation"] = results
        return results

    # ------------------------------------------------------------------ #
    # Internal utilities
    # ------------------------------------------------------------------ #
    def _prepare_datasets(self) -> DatasetBundle:
        (
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            _,
        ) = prepare_compression_datasets()

        def build_ds(x, y, augment=False, is_training=False):
            ds = tf.data.Dataset.from_tensor_slices((x, y))
            if is_training:
                ds = ds.shuffle(10000)

            if augment:
                aug = tf.keras.Sequential(
                    [
                        tf.keras.layers.RandomFlip("horizontal"),
                        tf.keras.layers.RandomTranslation(0.1, 0.1),
                        tf.keras.layers.RandomRotation(0.05),
                    ]
                )

                def apply_aug(img, label):
                    return aug(img, training=True), label

                ds = ds.map(apply_aug, num_parallel_calls=AUTOTUNE)
            ds = ds.map(lambda img, label: (tf.cast(img, tf.float32), label), num_parallel_calls=AUTOTUNE)
            
            # CRITICAL: Always use drop_remainder=True to ensure consistent batch shapes
            # This prevents XLA JIT compilation errors and excessive retracing
            ds = ds.batch(self.batch_size, drop_remainder=True)
            
            if is_training:
                ds = ds.repeat()
                
            return ds.prefetch(AUTOTUNE)

        bundle = DatasetBundle(
            train=build_ds(x_train, y_train, augment=True, is_training=True),
            val=build_ds(x_val, y_val),
            test=build_ds(x_test, y_test),
            train_size=len(x_train),
            val_size=len(x_val),
            test_size=len(x_test),
        )

        return bundle

    def _get_dataset(self, split: str) -> tf.data.Dataset:
        if self._dataset_bundle is None:
            self._dataset_bundle = self._prepare_datasets()
        if split == "train":
            return self._dataset_bundle.train
        if split == "val":
            return self._dataset_bundle.val
        return self._dataset_bundle.test

    def _build_student(self, width_multiplier: float) -> tf.keras.Model:
        with _precision_guard("float32"):
            student = self.student_arch(width_multiplier)
        weights = [np.asarray(weight).astype(np.float32) for weight in student.get_weights()]
        if weights:
            student.set_weights(weights)
        return student

    def _analyze_soft_targets(self, temps: Sequence[float]) -> Dict[str, float]:
        x_val, y_val = next(iter(self._get_dataset("val").take(1)))
        logits = self.teacher(x_val, training=False)
        analysis = {}
        for temperature in temps:
            softened = tf.nn.softmax(logits / temperature, axis=-1)
            entropy = tf.reduce_mean(
                tf.reduce_sum(-softened * tf.math.log(softened + 1e-8), axis=-1)
            )
            analysis[str(float(temperature))] = float(entropy)
        return analysis

    def _default_attention_layers(self, model: tf.keras.Model) -> List[str]:
        conv_layers = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        return conv_layers[-3:] if len(conv_layers) >= 3 else conv_layers

    def _attention_loss(
        self,
        teacher_outputs: Sequence[tf.Tensor],
        student_outputs: Sequence[tf.Tensor],
    ) -> tf.Tensor:
        losses = []
        for teacher_feat, student_feat in zip(teacher_outputs, student_outputs):
            teacher_map = tf.reduce_mean(tf.square(teacher_feat), axis=-1)
            student_map = tf.reduce_mean(tf.square(student_feat), axis=-1)
            teacher_map = tf.nn.l2_normalize(tf.reshape(teacher_map, [tf.shape(teacher_map)[0], -1]), axis=-1)
            student_map = tf.nn.l2_normalize(tf.reshape(student_map, [tf.shape(student_map)[0], -1]), axis=-1)
            losses.append(tf.reduce_mean(tf.square(teacher_map - student_map)))
        return tf.add_n(losses) / max(1, len(losses))

    def _feature_loss(
        self,
        teacher_feats: Sequence[tf.Tensor],
        student_feats: Sequence[tf.Tensor],
    ) -> tf.Tensor:
        losses = []
        for teacher_feat, student_feat in zip(teacher_feats, student_feats):
            # Align channels if needed
            t_channels = teacher_feat.shape[-1]
            s_channels = student_feat.shape[-1]
            
            if t_channels != s_channels:
                # Project student features to match teacher channels
                student_feat = tf.keras.layers.Conv2D(
                    t_channels, 
                    kernel_size=1, 
                    padding='same',
                    use_bias=False
                )(student_feat)
            
            # Align spatial dimensions if needed
            t_shape = tf.shape(teacher_feat)
            s_shape = tf.shape(student_feat)
            if teacher_feat.shape[1] != student_feat.shape[1] or teacher_feat.shape[2] != student_feat.shape[2]:
                student_feat = tf.image.resize(
                    student_feat,
                    [t_shape[1], t_shape[2]],
                    method='bilinear'
                )
            
            teacher_norm = tf.nn.l2_normalize(teacher_feat, axis=-1)
            student_norm = tf.nn.l2_normalize(student_feat, axis=-1)
            losses.append(tf.reduce_mean(tf.square(teacher_norm - student_norm)))
        return tf.add_n(losses) / max(1, len(losses))

    def _noisy_teacher_copy(self, scale: float) -> tf.keras.Model:
        clone = self._clone_model_float32(self.teacher)
        for layer in clone.layers:
            weights = layer.get_weights()
            if not weights:
                continue
            noisy = [w + np.random.normal(scale=scale, size=w.shape) for w in weights]
            layer.set_weights(noisy)
        clone.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        return clone
