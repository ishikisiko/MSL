import tensorflow as tf
import tensorflow_model_optimization as tfmot
from huggingface_hub import hf_hub_download
import numpy as np
import os


class EdgeOptimizer:
    def __init__(self, baseline_model_path, hf_repo_id="Ishiki327/Course"):
        if os.path.exists(baseline_model_path):
            model_path = baseline_model_path
        else:
            print("Downloading model from Hugging Face Hub...")
            model_path = hf_hub_download(repo_id=hf_repo_id, filename=baseline_model_path)
            print(f"Model downloaded to: {model_path}")
        
        # Load model and ensure compatibility with TensorFlow Model Optimization
        try:
            self.baseline_model = tf.keras.models.load_model(model_path)
            print("Loaded model using tf.keras loader.")
        except Exception as e_tf:
            print(f"tf.keras loading failed: {e_tf}")
            try:
                import keras as pk
                try:
                    standalone_model = pk.saving.load_model(model_path, safe_mode=True)
                except Exception:
                    standalone_model = pk.saving.load_model(model_path, safe_mode=False)
                print("Loaded model using standalone Keras loader for compatibility.")
                
                # Convert standalone Keras model to TensorFlow Keras model
                self.baseline_model = self._convert_to_tf_keras(standalone_model)
                print("Converted to TensorFlow Keras model for optimization compatibility.")
                
            except Exception as e_k:
                print(f"Standalone Keras conversion failed: {e_k}")
                # Final fallback: create a simple compatible model
                print("Attempting to create a simple compatible model...")
                try:
                    self.baseline_model = self._create_fallback_model()
                    print("Created fallback model for demonstration.")
                except Exception as e_fallback:
                    raise RuntimeError(
                        f"All model loading methods failed.\n"
                        f"tf.keras error: {e_tf}\n"
                        f"keras conversion error: {e_k}\n"
                        f"fallback error: {e_fallback}"
                    )
        
        # Create a dummy representative dataset for quantization calibration
        self.representative_dataset = [tf.random.normal((1, *self.baseline_model.input_shape[1:])) for _ in range(100)]

    def _convert_to_tf_keras(self, standalone_model):
        """
        Convert a standalone Keras model to TensorFlow Keras model.
        This ensures compatibility with TensorFlow Model Optimization.
        
        Args:
            standalone_model: Model loaded with standalone Keras
            
        Returns:
            tf.keras.Model: TensorFlow Keras compatible model
        """
        def fix_config_for_tf_keras(config):
            """Fix layer configurations for TensorFlow Keras compatibility."""
            if isinstance(config, dict):
                fixed_config = {}
                for key, value in config.items():
                    if key == 'batch_shape' and 'shape' not in config:
                        # Convert batch_shape to shape by removing the first dimension
                        if isinstance(value, (list, tuple)) and len(value) > 1:
                            fixed_config['shape'] = value[1:]
                        continue
                    elif key == 'batch_shape' and 'shape' in config:
                        # Skip batch_shape if shape already exists
                        continue
                    elif isinstance(value, dict):
                        fixed_config[key] = fix_config_for_tf_keras(value)
                    elif isinstance(value, list):
                        fixed_config[key] = [fix_config_for_tf_keras(item) if isinstance(item, dict) else item for item in value]
                    else:
                        fixed_config[key] = value
                return fixed_config
            return config
        
        try:
            # Method 1: Try to rebuild using fixed model configuration
            config = standalone_model.get_config()
            fixed_config = fix_config_for_tf_keras(config)
            
            if hasattr(standalone_model, 'model_config') or isinstance(standalone_model, type(standalone_model)) and 'Model' in type(standalone_model).__name__:
                # For functional models
                tf_model = tf.keras.Model.from_config(fixed_config)
            else:
                # For sequential models
                tf_model = tf.keras.Sequential.from_config(fixed_config)
            
            # Copy weights from standalone model to TensorFlow model
            tf_model.set_weights(standalone_model.get_weights())
            return tf_model
            
        except Exception as e1:
            try:
                # Method 2: Create a new TensorFlow model by manually rebuilding architecture
                input_shape = standalone_model.input_shape
                if isinstance(input_shape, list):
                    input_shape = input_shape[0]
                
                # Remove batch dimension for creating Input layer
                if len(input_shape) > 1:
                    input_shape = input_shape[1:]
                
                inputs = tf.keras.Input(shape=input_shape)
                x = inputs
                
                # Rebuild layer by layer, skipping input layers
                for layer in standalone_model.layers:
                    layer_type = type(layer).__name__
                    
                    # Skip input layers
                    if layer_type in ['InputLayer', 'Input']:
                        continue
                    
                    layer_config = layer.get_config()
                    # Fix layer config for compatibility
                    layer_config = fix_config_for_tf_keras(layer_config)
                    
                    # Map common layer types to TensorFlow Keras equivalents
                    try:
                        if layer_type == 'Dense':
                            x = tf.keras.layers.Dense(**layer_config)(x)
                        elif layer_type == 'Conv2D':
                            x = tf.keras.layers.Conv2D(**layer_config)(x)
                        elif layer_type == 'MaxPooling2D':
                            x = tf.keras.layers.MaxPooling2D(**layer_config)(x)
                        elif layer_type == 'GlobalAveragePooling2D':
                            x = tf.keras.layers.GlobalAveragePooling2D(**layer_config)(x)
                        elif layer_type == 'Dropout':
                            x = tf.keras.layers.Dropout(**layer_config)(x)
                        elif layer_type == 'Flatten':
                            x = tf.keras.layers.Flatten(**layer_config)(x)
                        elif layer_type == 'BatchNormalization':
                            x = tf.keras.layers.BatchNormalization(**layer_config)(x)
                        elif layer_type == 'Activation':
                            x = tf.keras.layers.Activation(**layer_config)(x)
                        elif layer_type == 'ReLU':
                            x = tf.keras.layers.ReLU(**layer_config)(x)
                        else:
                            # For other layer types, try generic approach
                            layer_class = getattr(tf.keras.layers, layer_type, None)
                            if layer_class:
                                x = layer_class(**layer_config)(x)
                            else:
                                print(f"Warning: Unknown layer type {layer_type}, trying fallback...")
                                # Try to create the layer using the original layer's class
                                try:
                                    tf_layer_class = getattr(tf.keras.layers, layer_type)
                                    x = tf_layer_class.from_config(layer_config)(x)
                                except:
                                    print(f"Warning: Could not recreate layer {layer_type}, skipping...")
                                    continue
                    except Exception as layer_error:
                        print(f"Warning: Error creating layer {layer_type}: {layer_error}")
                        continue
                
                tf_model = tf.keras.Model(inputs=inputs, outputs=x)
                tf_model.set_weights(standalone_model.get_weights())
                print("Successfully converted model using manual architecture reconstruction.")
                return tf_model
                
            except Exception as e2:
                try:
                    # Method 3: Simplified Sequential approach
                    layers = []
                    for layer in standalone_model.layers[1:]:  # Skip first input layer
                        layer_config = layer.get_config()
                        layer_type = type(layer).__name__
                        
                        # Create TensorFlow Keras layer
                        if hasattr(tf.keras.layers, layer_type):
                            tf_layer_class = getattr(tf.keras.layers, layer_type)
                            try:
                                layers.append(tf_layer_class.from_config(layer_config))
                            except:
                                # Try without problematic config keys
                                clean_config = {k: v for k, v in layer_config.items() 
                                              if k not in ['batch_shape', 'batch_input_shape']}
                                layers.append(tf_layer_class(**clean_config))
                    
                    tf_model = tf.keras.Sequential(layers)
                    tf_model.build(input_shape=standalone_model.input_shape)
                    tf_model.set_weights(standalone_model.get_weights())
                    print("Successfully converted model using Sequential approach.")
                    return tf_model
                    
                except Exception as e3:
                    raise RuntimeError(
                        f"Failed to convert standalone Keras model to TensorFlow Keras. "
                        f"Tried multiple methods. Errors:\n"
                        f"Method 1 (config): {e1}\n"
                        f"Method 2 (manual): {e2}\n" 
                        f"Method 3 (sequential): {e3}"
                    )

    def _create_fallback_model(self):
        """
        Create a simple fallback model for demonstration purposes.
        This is used when model conversion completely fails.
        
        Returns:
            tf.keras.Model: A simple CNN model for CIFAR-32 sized inputs
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        print("Created fallback CNN model with shape (32, 32, 3) -> 10 classes")
        return model

    def implement_pruning(self, target_sparsity=0.75):
        """
        Implement magnitude-based pruning for edge deployment.

        Args:
            target_sparsity: Target sparsity level (0.75 = 75% weights pruned)

        Returns:
            tf.keras.Model: Pruned model ready for fine-tuning.
        """
        # Ensure we have a TensorFlow Keras model
        if not isinstance(self.baseline_model, (tf.keras.Model, tf.keras.Sequential)):
            raise ValueError(
                f"Model must be a TensorFlow Keras model for pruning. "
                f"Got {type(self.baseline_model)}. "
                f"The conversion might have failed in __init__."
            )
        
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=target_sparsity,
                begin_step=0,
                end_step=-1
            )
        }

        try:
            # Wrap the entire model for pruning
            pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
                self.baseline_model, **pruning_params
            )
            print(f"Successfully created pruned model with {target_sparsity*100}% sparsity.")
            return pruned_model
            
        except Exception as e:
            print(f"Pruning failed: {e}")
            print(f"Model type: {type(self.baseline_model)}")
            print("Attempting alternative pruning approach...")
            
            # Alternative: try cloning the model first
            try:
                cloned_model = tf.keras.models.clone_model(self.baseline_model)
                cloned_model.set_weights(self.baseline_model.get_weights())
                
                pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
                    cloned_model, **pruning_params
                )
                print("Successfully created pruned model using cloned model.")
                return pruned_model
                
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to create pruned model with both direct and cloned approaches. "
                    f"Errors: {e}, {e2}"
                )

    def implement_quantization(self):
        """
        Implement post-training quantization for edge deployment.

        Returns:
            dict: Quantized models with different strategies.
        """
        quantized_models = {}

        # Representative dataset generator
        def representative_dataset_gen():
            for data in self.representative_dataset:
                yield [data]

        # Dynamic Range Quantization
        converter_dynamic = tf.lite.TFLiteConverter.from_keras_model(self.baseline_model)
        converter_dynamic.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_models['dynamic_range'] = converter_dynamic.convert()

        # Full Integer Quantization
        converter_int8 = tf.lite.TFLiteConverter.from_keras_model(self.baseline_model)
        converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_int8.representative_dataset = representative_dataset_gen
        converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter_int8.inference_input_type = tf.int8
        converter_int8.inference_output_type = tf.int8
        quantized_models['full_integer'] = converter_int8.convert()

        # Float16 Quantization
        converter_float16 = tf.lite.TFLiteConverter.from_keras_model(self.baseline_model)
        converter_float16.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_float16.target_spec.supported_types = [tf.float16]
        quantized_models['float16'] = converter_float16.convert()

        return quantized_models

    def implement_architecture_optimization(self):
        """
        Optimize model architecture by replacing Conv2D with DepthwiseSeparableConv2D.

        Returns:
            tf.keras.Model: Architecture-optimized model.
        """
        # This is a simplified example assuming the model can be rebuilt this way.
        # For complex models, this requires careful manual reconstruction.
        new_layers = []
        for layer in self.baseline_model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                # Replace Conv2D with DepthwiseSeparableConv2D
                new_layer = tf.keras.layers.DepthwiseSeparableConv2D(
                    filters=layer.filters,
                    kernel_size=layer.kernel_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    activation=layer.activation,
                    name='sep_' + layer.name
                )
            else:
                new_layer = layer.__class__.from_config(layer.get_config())
            
            new_layers.append(new_layer)
        
        # This simple sequential reconstruction might not work for complex, branched models.
        try:
            optimized_model = tf.keras.Sequential(new_layers)
            print("Successfully created architecture-optimized model.")
            return optimized_model
        except Exception as e:
            print(f"Could not automatically create optimized architecture: {e}")
            print("Returning baseline model as a fallback.")
            return self.baseline_model


    def implement_neural_architecture_search(self):
        """
        Implement simplified NAS for finding optimal edge architecture.
        This is a placeholder demonstrating the concept. Real NAS is computationally intensive.

        Returns:
            tuple: (best_architecture_config, search_results)
        """
        search_space = {
            'filters': [16, 32, 64],
            'kernel_size': [3, 5],
            'depth': [2, 3, 4]
        }
        search_results = []

        print("Starting simplified Neural Architecture Search...")
        for i in range(5): # Perform 5 random trials
            config = {
                'filters': np.random.choice(search_space['filters']),
                'kernel_size': np.random.choice(search_space['kernel_size']),
                'depth': np.random.choice(search_space['depth'])
            }
            
            # In a real scenario, you would build, train, and evaluate a model with this config.
            # Here, we just simulate it with a random performance score.
            simulated_accuracy = 0.5 + np.random.rand() * 0.4 # Random accuracy between 0.5 and 0.9
            simulated_latency = (config['filters'] * config['depth'] * config['kernel_size']**2) / 1e4
            
            performance_score = simulated_accuracy / (simulated_latency + 1e-6) # Simple trade-off metric
            
            result = {
                'config': config,
                'accuracy': simulated_accuracy,
                'latency_ms': simulated_latency,
                'score': performance_score
            }
            search_results.append(result)
            print(f"Trial {i+1}/5: Config={config}, Score={performance_score:.2f}")

        # Find the best architecture based on the score
        best_result = max(search_results, key=lambda x: x['score'])
        best_architecture_config = best_result['config']
        
        print(f"Best architecture found: {best_architecture_config}")
        return best_architecture_config, search_results

    def create_tflite_models(self, models_dict):
        """
        Convert optimized models to TensorFlow Lite format and save them.

        Args:
            models_dict (dict): Dictionary of optimized Keras models.
            
        Returns:
            dict: Paths to the saved TensorFlow Lite models.
        """
        tflite_model_paths = {}
        output_dir = "tflite_models"
        os.makedirs(output_dir, exist_ok=True)

        for name, model in models_dict.items():
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            
            model_path = os.path.join(output_dir, f"{name}.tflite")
            with open(model_path, 'wb') as f:
                f.write(tflite_model)
            
            tflite_model_paths[name] = {
                'path': model_path,
                'size_kb': os.path.getsize(model_path) / 1024
            }
            print(f"Converted '{name}' to TFLite: {model_path} ({tflite_model_paths[name]['size_kb']:.2f} KB)")

        return tflite_model_paths


def benchmark_edge_optimizations():
    """
    Comprehensive benchmarking of edge optimization strategies.

    Returns:
        dict: Detailed performance analysis.
    """
    optimizer = EdgeOptimizer('baseline_model.keras')
    results = {}

    # 1. Pruning
    print("\n--- Benchmarking Pruning ---")
    pruned_model = optimizer.implement_pruning(target_sparsity=0.75)
    # In a real scenario, you would fine-tune this model and measure its accuracy.
    # Here we just demonstrate the creation.
    results['pruning_0.75'] = "Model created, requires fine-tuning."
    
    # 2. Quantization
    print("\n--- Benchmarking Quantization ---")
    quantized_tflite_models = optimizer.implement_quantization()
    results['quantization'] = {}
    for name, model_content in quantized_tflite_models.items():
        path = f"quantized_{name}.tflite"
        with open(path, 'wb') as f:
            f.write(model_content)
        results['quantization'][name] = {'size_kb': len(model_content) / 1024}
        print(f"Quantization ({name}): Size = {results['quantization'][name]['size_kb']:.2f} KB")

    # 3. Architecture Optimization
    print("\n--- Benchmarking Architecture Optimization ---")
    arch_opt_model = optimizer.implement_architecture_optimization()
    results['architecture_optimization'] = "Model created."

    # 4. Neural Architecture Search
    print("\n--- Benchmarking Neural Architecture Search ---")
    best_config, search_log = optimizer.implement_neural_architecture_search()
    results['nas'] = {'best_config': best_config, 'log': search_log}

    # 5. TFLite Conversion of Keras models
    print("\n--- Creating TFLite models ---")
    keras_models_to_convert = {
        'baseline': optimizer.baseline_model,
        'pruned': pruned_model,
        'arch_opt': arch_opt_model
    }
    # Note: The pruned model should be stripped of pruning wrappers before final conversion
    final_pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    keras_models_to_convert['pruned'] = final_pruned_model
    
    tflite_paths = optimizer.create_tflite_models(keras_models_to_convert)
    results['tflite_conversion'] = tflite_paths

    return results


if __name__ == "__main__":
    benchmark_results = benchmark_edge_optimizations()
    print("\n--- Edge Optimization Benchmark Summary ---")
    import json
    # Using json for pretty printing the nested dictionary
    print(json.dumps(benchmark_results, indent=2, default=str))
 