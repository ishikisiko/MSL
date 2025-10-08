import tensorflow as tf 

import json 

import numpy as np 
 
import os 

from abc import ABC, abstractmethod 

from dataclasses import dataclass 
from dataclasses import asdict 

from typing import Dict, List, Tuple, Any

import psutil
import gc

try:
    from huggingface_hub import hf_hub_download, list_repo_files
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Warning: huggingface_hub not available. Only local model loading supported.") 
 
 
# =============================
# Input/Output path definitions
# =============================
BASELINE_MODEL_PATH = 'baseline_model.keras'
ARTIFACTS_DIR = 'models'
RESULTS_DIR = 'results'
REPORT_PATH = os.path.join(RESULTS_DIR, 'multi_scale_optimization_report.json')
 
 
def ensure_directory_exists(directory_path: str) -> None:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
 
 
def get_file_size_mb(file_path: str) -> float:
    try:
        size_bytes = os.path.getsize(file_path)
        return float(size_bytes) / (1024.0 * 1024.0)
    except OSError:
        return 0.0
 
 
def estimate_latency_ms(parameter_count: int, scale_factor: float) -> float:
    baseline_ms = max(1.0, parameter_count / 1_000_000.0)
    return baseline_ms * scale_factor


def measure_memory_usage_mb(model_path: str, model_type: str = 'keras', 
                            x_test: np.ndarray = None) -> float:
    """
    Measure actual memory usage of a model during inference.
    
    Args:
        model_path: Path to the model file
        model_type: 'keras' or 'tflite'
        x_test: Test data for inference (uses first 10 samples if provided)
        
    Returns:
        Memory usage in MB
    """
    try:
        import psutil
        process = psutil.Process()
        
        # Force garbage collection before measurement
        gc.collect()
        
        # Measure baseline memory
        baseline_memory = process.memory_info().rss / (1024 * 1024)
        
        # Load model and measure memory
        if model_type == 'keras':
            model = tf.keras.models.load_model(model_path)
            
            # Run inference if test data provided
            if x_test is not None:
                test_sample = x_test[:min(10, len(x_test))]
                _ = model.predict(test_sample, verbose=0)
            
            # Measure peak memory after loading and inference
            peak_memory = process.memory_info().rss / (1024 * 1024)
            
            # Clean up
            del model
            
        elif model_type == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            # Run inference if test data provided
            if x_test is not None:
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                for i in range(min(10, len(x_test))):
                    input_data = x_test[i:i+1].astype(np.float32)
                    
                    # Handle quantized input
                    if input_details[0]['dtype'] == np.int8:
                        input_scale, input_zero_point = input_details[0]['quantization']
                        if input_scale > 0:
                            input_data = (input_data / input_scale + input_zero_point).astype(np.int8)
                        else:
                            input_data = ((input_data - 0.5) * 255).astype(np.int8)
                    
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    _ = interpreter.get_tensor(output_details[0]['index'])
            
            # Measure peak memory
            peak_memory = process.memory_info().rss / (1024 * 1024)
            
            # Clean up
            del interpreter
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Force garbage collection after measurement
        gc.collect()
        
        # Calculate actual memory used
        memory_used = peak_memory - baseline_memory
        
        # Return measured memory, with a minimum based on model file size
        model_size = get_file_size_mb(model_path)
        return max(memory_used, model_size * 1.2)
        
    except ImportError:
        print("Warning: psutil not installed. Using estimated memory.")
        # Fallback to estimation
        model_size = get_file_size_mb(model_path)
        if model_type == 'keras':
            return max(model_size * 2.0, 256.0)
        elif model_type == 'tflite':
            return max(model_size * 1.5, 8.0)
        return model_size * 2.0
    except Exception as e:
        print(f"Warning: Failed to measure memory usage: {e}. Using estimation.")
        # Fallback to estimation
        model_size = get_file_size_mb(model_path)
        if model_type == 'keras':
            return max(model_size * 2.0, 256.0)
        elif model_type == 'tflite':
            return max(model_size * 1.5, 8.0)
        return model_size * 2.0
 
 
def representative_data_gen_for_model(model: tf.keras.Model, num_samples: int = 100):
    if not hasattr(model, 'inputs') or model.inputs is None:
        return
    input_specs = []
    for tensor in model.inputs:
        shape = [dim if dim is not None else 1 for dim in tensor.shape]
        if len(shape) > 0:
            shape[0] = 1
        dtype = tensor.dtype if hasattr(tensor, 'dtype') else tf.float32
        input_specs.append((shape, dtype))
    for _ in range(num_samples):
        sample = []
        for shape, dtype in input_specs:
            np_dtype = np.float32 if getattr(dtype, 'is_floating', True) else np.int8
            sample.append(np.random.uniform(-1.0, 1.0, size=shape).astype(np_dtype))
        if len(sample) == 1:
            yield [sample[0]]
        else:
            yield sample


def load_test_data(num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-10 test data for model evaluation."""
    try:
        (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_test = x_test.astype('float32') / 255.0
        y_test = y_test.astype('int32').squeeze()
        
        # Limit to num_samples for faster evaluation
        if num_samples > 0 and num_samples < len(x_test):
            x_test = x_test[:num_samples]
            y_test = y_test[:num_samples]
        
        return x_test, y_test
    except Exception as e:
        print(f"Warning: Failed to load CIFAR-10 data: {e}")
        return None, None


def evaluate_keras_model(model: tf.keras.Model, x_test: np.ndarray, y_test: np.ndarray) -> float:
    """Evaluate a Keras model and return accuracy."""
    try:
        if x_test is None or y_test is None:
            return 0.0
        
        predictions = model.predict(x_test, verbose=0, batch_size=128)
        predicted_classes = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_classes == y_test)
        return float(accuracy)
    except Exception as e:
        print(f"Warning: Failed to evaluate Keras model: {e}")
        return 0.0


def evaluate_tflite_model(model_path: str, x_test: np.ndarray, y_test: np.ndarray) -> float:
    """Evaluate a TFLite model and return accuracy."""
    try:
        if x_test is None or y_test is None:
            return 0.0
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Get input properties
        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']
        
        correct = 0
        total = len(y_test)
        
        # Evaluate each sample
        for i in range(total):
            # Prepare input
            input_data = x_test[i:i+1].astype(np.float32)
            
            # Convert to int8 if needed
            if input_dtype == np.int8:
                input_scale, input_zero_point = input_details[0]['quantization']
                if input_scale > 0:
                    input_data = (input_data / input_scale + input_zero_point).astype(np.int8)
                else:
                    input_data = ((input_data - 0.5) * 255).astype(np.int8)
            
            # Run inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # Convert output if quantized
            if output_details[0]['dtype'] == np.int8:
                output_scale, output_zero_point = output_details[0]['quantization']
                if output_scale > 0:
                    output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
            
            # Get prediction
            predicted_class = np.argmax(output_data)
            if predicted_class == y_test[i]:
                correct += 1
        
        accuracy = correct / total
        return float(accuracy)
    except Exception as e:
        print(f"Warning: Failed to evaluate TFLite model: {e}")
        return 0.0

 

@dataclass 

class DeploymentTarget: 

    """Configuration for different deployment targets.""" 

    name: str 

    max_model_size_mb: float 

    max_latency_ms: float 

    max_memory_mb: float 

    power_budget_mw: float 

    compute_capability: str  # 'cloud', 'edge', 'tiny' 

 

@dataclass 

class OptimizationResult: 

    """Results from model optimization.""" 

    model_path: str 

    accuracy: float 

    model_size_mb: float 

    estimated_latency_ms: float 

    memory_usage_mb: float 

    optimization_strategy: str 

 

class ModelOptimizer(ABC): 

    @abstractmethod 

    def optimize(self, model: tf.keras.Model, target: DeploymentTarget, 
                 x_test: np.ndarray = None, y_test: np.ndarray = None) -> OptimizationResult: 

        pass 

 

class CloudOptimizer(ModelOptimizer): 

    def optimize(self, model: tf.keras.Model, target: DeploymentTarget,
                 x_test: np.ndarray = None, y_test: np.ndarray = None) -> OptimizationResult: 
        ensure_directory_exists(ARTIFACTS_DIR)
        output_path = os.path.join(ARTIFACTS_DIR, f"{target.name}_optimized.keras")
        model.save(output_path)

        params = int(model.count_params())
        size_mb = get_file_size_mb(output_path)
        estimated_latency = estimate_latency_ms(params, scale_factor=5.0)
        
        # Measure actual memory usage
        print(f"Measuring {target.name} memory usage...")
        measured_memory_mb = measure_memory_usage_mb(output_path, model_type='keras', x_test=x_test)
        print(f"  Measured memory: {measured_memory_mb:.2f} MB")
        
        # Evaluate model accuracy
        print(f"Evaluating {target.name} model accuracy...")
        accuracy = evaluate_keras_model(model, x_test, y_test)
        print(f"  Accuracy: {accuracy:.4f}")

        return OptimizationResult(
            model_path=output_path,
            accuracy=accuracy,
            model_size_mb=size_mb,
            estimated_latency_ms=estimated_latency,
            memory_usage_mb=measured_memory_mb,
            optimization_strategy='keras_fp32_saving'
        ) 

 

class EdgeOptimizer(ModelOptimizer): 

    def optimize(self, model: tf.keras.Model, target: DeploymentTarget,
                 x_test: np.ndarray = None, y_test: np.ndarray = None) -> OptimizationResult: 
        ensure_directory_exists(ARTIFACTS_DIR)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        try:
            tflite_model = converter.convert()
        except Exception:
            converter.optimizations = []
            tflite_model = converter.convert()

        output_path = os.path.join(ARTIFACTS_DIR, f"{target.name}_dynamic.tflite")
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        params = int(model.count_params())
        size_mb = get_file_size_mb(output_path)
        estimated_latency = estimate_latency_ms(params, scale_factor=20.0)
        
        # Measure actual memory usage
        print(f"Measuring {target.name} memory usage...")
        measured_memory_mb = measure_memory_usage_mb(output_path, model_type='tflite', x_test=x_test)
        print(f"  Measured memory: {measured_memory_mb:.2f} MB")
        
        # Evaluate model accuracy
        print(f"Evaluating {target.name} model accuracy...")
        accuracy = evaluate_tflite_model(output_path, x_test, y_test)
        print(f"  Accuracy: {accuracy:.4f}")

        return OptimizationResult(
            model_path=output_path,
            accuracy=accuracy,
            model_size_mb=size_mb,
            estimated_latency_ms=estimated_latency,
            memory_usage_mb=measured_memory_mb,
            optimization_strategy='tflite_dynamic_range'
        ) 

 

class TinyMLOptimizer(ModelOptimizer): 

    def optimize(self, model: tf.keras.Model, target: DeploymentTarget,
                 x_test: np.ndarray = None, y_test: np.ndarray = None) -> OptimizationResult: 
        ensure_directory_exists(ARTIFACTS_DIR)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_data_gen_for_model(model, num_samples=100)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        try:
            tflite_model = converter.convert()
            strategy = 'tflite_full_int8'
            filename = f"{target.name}_int8.tflite"
        except Exception:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            strategy = 'tflite_dynamic_range_fallback'
            filename = f"{target.name}_dynamic.tflite"

        output_path = os.path.join(ARTIFACTS_DIR, filename)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        params = int(model.count_params())
        size_mb = get_file_size_mb(output_path)
        estimated_latency = estimate_latency_ms(params, scale_factor=50.0)
        
        # Measure actual memory usage
        print(f"Measuring {target.name} memory usage...")
        measured_memory_mb = measure_memory_usage_mb(output_path, model_type='tflite', x_test=x_test)
        print(f"  Measured memory: {measured_memory_mb:.2f} MB")
        
        # Evaluate model accuracy
        print(f"Evaluating {target.name} model accuracy...")
        accuracy = evaluate_tflite_model(output_path, x_test, y_test)
        print(f"  Accuracy: {accuracy:.4f}")

        return OptimizationResult(
            model_path=output_path,
            accuracy=accuracy,
            model_size_mb=size_mb,
            estimated_latency_ms=estimated_latency,
            memory_usage_mb=measured_memory_mb,
            optimization_strategy=strategy
        ) 

 

class MultiScaleDeploymentPipeline: 

    """ 

    Automated pipeline for optimizing models across different deployment scales. 

    """ 

     

    def __init__(self, hf_repo_id="Ishiki327/Course"):
        self.hf_repo_id = hf_repo_id 

        self.optimizers = { 

            'cloud': CloudOptimizer(), 

            'edge': EdgeOptimizer(), 

            'tiny': TinyMLOptimizer() 

        } 

         

        # Define deployment targets 

        self.targets = { 

            'cloud_server': DeploymentTarget( 

                name='cloud_server', 

                max_model_size_mb=1000.0, 

                max_latency_ms=100.0, 

                max_memory_mb=8000.0, 

                power_budget_mw=50000.0, 

                compute_capability='cloud' 

            ), 

            'edge_device': DeploymentTarget( 

                name='edge_device', 

                max_model_size_mb=50.0, 

                max_latency_ms=200.0, 

                max_memory_mb=512.0, 

                power_budget_mw=2000.0, 

                compute_capability='edge' 

            ), 

            'microcontroller': DeploymentTarget( 

                name='microcontroller', 

                max_model_size_mb=1.0, 

                max_latency_ms=1000.0, 

                max_memory_mb=64.0, 

                power_budget_mw=10.0, 

                compute_capability='tiny' 

            ) 

        } 

    
    def _load_model(self, baseline_model_path: str) -> tf.keras.Model:
        """
        Load model from local path or download from Hugging Face Hub if not found locally.
        
        Args:
            baseline_model_path: Path to baseline model file
            
        Returns:
            Loaded Keras model
        """
        if os.path.exists(baseline_model_path):
            print(f"Loading model from local path: {baseline_model_path}")
            model_path = baseline_model_path
        else:
            if not HF_HUB_AVAILABLE:
                raise ImportError(
                    "huggingface_hub is not installed. Please install it with: pip install huggingface_hub"
                )
            print(f"Model not found locally. Downloading from Hugging Face Hub: {self.hf_repo_id}")
            
            # First, let's list available files in the repository
            try:
                print(f"Checking available files in repository {self.hf_repo_id}...")
                repo_files = list_repo_files(repo_id=self.hf_repo_id)
                print(f"Available files in repository:")
                for file in sorted(repo_files):
                    print(f"  - {file}")
                    
                # Look for model files
                model_files = [f for f in repo_files if f.endswith(('.keras', '.h5', '.pb', '.tflite'))]
                if model_files:
                    print(f"Found model files: {model_files}")
                    
                # Check if the requested file exists with different extensions or paths
                possible_files = [
                    baseline_model_path,
                    f"models/{baseline_model_path}",
                    f"checkpoints/{baseline_model_path}",
                    baseline_model_path.replace('.keras', '.h5'),
                    baseline_model_path.replace('.keras', '.pb'),
                ]
                
                found_file = None
                for possible_file in possible_files:
                    if possible_file in repo_files:
                        found_file = possible_file
                        print(f"Found model file: {found_file}")
                        break
                
                if found_file:
                    model_path = hf_hub_download(repo_id=self.hf_repo_id, filename=found_file)
                    print(f"Model downloaded to: {model_path}")
                else:
                    # If no exact match, suggest available model files
                    error_msg = f"Model file '{baseline_model_path}' not found in repository {self.hf_repo_id}.\n"
                    if model_files:
                        error_msg += f"Available model files: {', '.join(model_files)}\n"
                        error_msg += "Consider updating BASELINE_MODEL_PATH to one of the available files."
                    else:
                        error_msg += "No model files found in the repository."
                    raise FileNotFoundError(error_msg)
                    
            except Exception as e:
                if isinstance(e, FileNotFoundError):
                    raise e
                raise RuntimeError(f"Failed to access Hugging Face repository {self.hf_repo_id}: {e}")
        
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

     

    def optimize_for_all_targets(self, baseline_model_path: str) -> Dict[str, OptimizationResult]: 

        """ 

        Optimize baseline model for all deployment targets. 

         

        Args: 

            baseline_model_path: Path to baseline Keras model or filename in Hugging Face repo 

             

        Returns: 

            Dictionary mapping target names to optimization results 

        """ 

        baseline_model = self._load_model(baseline_model_path) 

        
        # Load test data for evaluation
        print("\nLoading test data for model evaluation...")
        x_test, y_test = load_test_data(num_samples=1000)
        if x_test is not None:
            print(f"Loaded {len(x_test)} test samples for evaluation.\n")
        else:
            print("Warning: Test data not available. Accuracy will be 0.0.\n")
        
        results = {} 

         

        for target_name, target_config in self.targets.items(): 

            print(f"\nOptimizing for {target_name}...")
            optimizer = self.optimizers[target_config.compute_capability] 

            results[target_name] = optimizer.optimize(baseline_model, target_config, x_test, y_test) 

             

        return results 

     

    def analyze_scaling_trade_offs(self, results: Dict[str, OptimizationResult]) -> Dict[str, Any]: 
        """ 
        
        Analyze trade-offs across different deployment scales. 
        
        
        Args: 
        
            results: Optimization results from optimize_for_all_targets 
        
            
        Returns: 
        
            Comprehensive analysis of scaling trade-offs 
        
        """ 
        analysis: Dict[str, Any] = {'targets': {}, 'best': {}} 
        
        for target_name, result in results.items(): 
            target_cfg = self.targets[target_name] 
            meets_size = result.model_size_mb <= target_cfg.max_model_size_mb 
            meets_latency = result.estimated_latency_ms <= target_cfg.max_latency_ms 
            meets_memory = result.memory_usage_mb <= target_cfg.max_memory_mb 
            feasible = bool(meets_size and meets_latency and meets_memory) 
            analysis['targets'][target_name] = { 
                'feasible': feasible, 
                'meets_size': meets_size, 
                'meets_latency': meets_latency, 
                'meets_memory': meets_memory, 
                'strategy': result.optimization_strategy, 
                'metrics': { 
                    'model_size_mb': result.model_size_mb, 
                    'estimated_latency_ms': result.estimated_latency_ms, 
                    'memory_usage_mb': result.memory_usage_mb, 
                    'accuracy': result.accuracy, 
                }, 
                'model_path': result.model_path, 
            } 
        
        feasible_items = [ 
            (name, info) for name, info in analysis['targets'].items() if info['feasible'] 
        ] 
        if feasible_items: 
            best_size = min(feasible_items, key=lambda x: x[1]['metrics']['model_size_mb']) 
            best_latency = min(feasible_items, key=lambda x: x[1]['metrics']['estimated_latency_ms']) 
            analysis['best']['smallest_model'] = best_size[0] 
            analysis['best']['lowest_latency'] = best_latency[0] 
        else: 
            analysis['best']['smallest_model'] = None 
            analysis['best']['lowest_latency'] = None 
        
        return analysis 

 

    def generate_deployment_recommendations(self, analysis: Dict[str, Any]) -> List[str]: 
        """ 
        
        Generate actionable deployment recommendations. 
        
        
        Args: 
        
            analysis: Results from analyze_scaling_trade_offs 
        
            
        Returns: 
        
            List of deployment recommendations 
        
        """ 
        recommendations: List[str] = [] 
        targets_info = analysis.get('targets', {}) 
        
        feasible = [name for name, info in targets_info.items() if info.get('feasible')] 
        if feasible: 
            recommendations.append(f"Feasible targets: {', '.join(feasible)}.") 
        else: 
            recommendations.append("No targets fully meet constraints; consider relaxing constraints or further compression.") 
        
        for name, info in targets_info.items(): 
            unmet = [] 
            if not info.get('meets_size'): 
                unmet.append('size') 
            if not info.get('meets_latency'): 
                unmet.append('latency') 
            if not info.get('meets_memory'): 
                unmet.append('memory') 
            if unmet: 
                hint = [] 
                if 'size' in unmet:
                    hint.append('pruning and more aggressive quantization') 
                if 'latency' in unmet:
                    hint.append('operator fusion and model distillation') 
                if 'memory' in unmet:
                    hint.append('reduced batch size and smaller intermediate activations') 
                recommendations.append(
                    f"For {name}, unmet constraints: {', '.join(unmet)}; consider {', '.join(hint)}."
                ) 
        
        best = analysis.get('best', {}) 
        if best.get('smallest_model'):
            recommendations.append(f"Smallest feasible model: {best['smallest_model']}.") 
        if best.get('lowest_latency'):
            recommendations.append(f"Lowest-latency feasible model: {best['lowest_latency']}.") 
        
        return recommendations 

 

def run_multi_scale_optimization(): 

    """ 

    Execute complete multi-scale optimization pipeline. 

    """ 

    pipeline = MultiScaleDeploymentPipeline() 

     

    # Optimize for all targets 

    results = pipeline.optimize_for_all_targets(BASELINE_MODEL_PATH) 

     

    # Analyze trade-offs 

    analysis = pipeline.analyze_scaling_trade_offs(results) 

     

    # Generate recommendations 

    recommendations = pipeline.generate_deployment_recommendations(analysis) 

     

    # Generate comprehensive report 

    serializable_results = {name: asdict(res) for name, res in results.items()} 

    report = { 

        'optimization_results': serializable_results, 

        'scaling_analysis': analysis, 

        'deployment_recommendations': recommendations 

    } 

     

    # Save report 

    ensure_directory_exists(RESULTS_DIR) 

    with open(REPORT_PATH, 'w') as f: 

        json.dump(report, f, indent=2) 

     

    return report 

 

if __name__ == "__main__": 

    report = run_multi_scale_optimization() 

    print("Multi-Scale Optimization Complete!") 

    print(f"Report saved to: {REPORT_PATH}") 