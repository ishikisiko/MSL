"""
Microcontroller Inference Script
==================================
This script demonstrates how to perform inference using the INT8 quantized TFLite model.
Suitable for microcontrollers (Arduino, ESP32, STM32) and ultra-low-power edge devices.

Requirements:
- TensorFlow Lite Runtime (or full TensorFlow)
- NumPy
- Minimal RAM (<64MB)

Note: This is a simulation script for testing the quantized model on a PC.
For actual microcontroller deployment, use TensorFlow Lite Micro (TFLM) in C++.

Usage:
    python inference_microcontroller.py --model models/microcontroller_int8.tflite --image path/to/image.png
    python inference_microcontroller.py --batch  # Run batch inference on test dataset
    python inference_microcontroller.py --benchmark  # Run latency benchmark
"""

import numpy as np
import argparse
import os
import time
from typing import Dict, List

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Trying TFLite Runtime...")
    try:
        import tflite_runtime.interpreter as tflite
        TFLITE_RUNTIME = True
    except ImportError:
        TFLITE_RUNTIME = False
        raise ImportError("Neither TensorFlow nor TFLite Runtime is available!")

# CIFAR-10 class names
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


class MicrocontrollerInference:
    """
    Wrapper class for INT8 quantized TFLite inference optimized for microcontrollers.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize TFLite interpreter for INT8 quantized model.
        
        Args:
            model_path: Path to the INT8 TFLite model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        print(f"Loading INT8 quantized TFLite model from: {model_path}")
        print("Note: This model is optimized for microcontrollers with limited resources")
        
        # Load interpreter
        if TF_AVAILABLE:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
        elif TFLITE_RUNTIME:
            self.interpreter = tflite.Interpreter(model_path=model_path)
        else:
            raise RuntimeError("No TFLite interpreter available!")
        
        # Allocate tensors
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Extract quantization parameters
        self.input_scale, self.input_zero_point = self.input_details[0]['quantization']
        self.output_scale, self.output_zero_point = self.output_details[0]['quantization']
        
        # Print model information
        self._print_model_info()
        
        # Calculate model size
        self.model_size_bytes = os.path.getsize(model_path)
        self.model_size_kb = self.model_size_bytes / 1024.0
        print(f"\nModel size: {self.model_size_kb:.2f} KB ({self.model_size_bytes} bytes)")
        print(f"Memory-efficient for microcontrollers: {'Yes' if self.model_size_kb < 1024 else 'No'}")
    
    def _print_model_info(self):
        """Print model input/output information."""
        print(f"\nModel loaded successfully!")
        print(f"\nInput details:")
        print(f"  Shape: {self.input_details[0]['shape']}")
        print(f"  Type: {self.input_details[0]['dtype']} (INT8 quantized)")
        print(f"  Quantization: scale={self.input_scale:.6f}, zero_point={self.input_zero_point}")
        
        print(f"\nOutput details:")
        print(f"  Shape: {self.output_details[0]['shape']}")
        print(f"  Type: {self.output_details[0]['dtype']} (INT8 quantized)")
        print(f"  Quantization: scale={self.output_scale:.6f}, zero_point={self.output_zero_point}")
    
    def quantize_input(self, image_array: np.ndarray) -> np.ndarray:
        """
        Quantize float32 input to INT8 for microcontroller inference.
        
        Args:
            image_array: Input image array (float32, normalized to [0, 1])
            
        Returns:
            Quantized INT8 array
        """
        if self.input_scale > 0:
            # Use learned quantization parameters
            quantized = (image_array / self.input_scale + self.input_zero_point)
            quantized = np.clip(quantized, -128, 127).astype(np.int8)
        else:
            # Fallback: simple quantization mapping [0, 1] to [-128, 127]
            quantized = ((image_array - 0.5) * 255)
            quantized = np.clip(quantized, -128, 127).astype(np.int8)
        
        return quantized
    
    def dequantize_output(self, output_data: np.ndarray) -> np.ndarray:
        """
        Dequantize INT8 output to float32.
        
        Args:
            output_data: Raw INT8 output from model
            
        Returns:
            Dequantized float32 array
        """
        if self.output_scale > 0:
            # Use learned quantization parameters
            dequantized = (output_data.astype(np.float32) - self.output_zero_point) * self.output_scale
        else:
            # Fallback: simple dequantization
            dequantized = output_data.astype(np.float32) / 127.0
        
        return dequantized
    
    def predict(self, image_array: np.ndarray, measure_time: bool = False) -> tuple:
        """
        Run INT8 inference on input image.
        
        Args:
            image_array: Input image (shape: [1, 32, 32, 3], normalized to [0, 1])
            measure_time: Whether to measure inference time
            
        Returns:
            Tuple of (predictions, inference_time_ms if measure_time else None)
        """
        # Quantize input
        input_data = self.quantize_input(image_array)
        
        if measure_time:
            start_time = time.perf_counter()
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference (this is what happens on the microcontroller)
        self.interpreter.invoke()
        
        # Get output tensor
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        if measure_time:
            inference_time = (time.perf_counter() - start_time) * 1000  # ms
        else:
            inference_time = None
        
        # Dequantize output
        output_data = self.dequantize_output(output_data)
        
        return output_data, inference_time
    
    def get_memory_info(self) -> Dict[str, any]:
        """
        Get memory usage information.
        
        Returns:
            Dictionary with memory usage details
        """
        # Get tensor arena size (approximate)
        input_size = np.prod(self.input_details[0]['shape'])
        output_size = np.prod(self.output_details[0]['shape'])
        
        # Estimate activation memory (very rough estimate)
        activation_memory_bytes = (input_size + output_size) * 1  # INT8 = 1 byte
        
        return {
            'model_size_kb': self.model_size_kb,
            'model_size_bytes': self.model_size_bytes,
            'estimated_activation_memory_kb': activation_memory_bytes / 1024.0,
            'total_estimated_memory_kb': self.model_size_kb + (activation_memory_bytes / 1024.0)
        }


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess a single image for inference.
    
    Args:
        image_path: Path to input image
        
    Returns:
        Preprocessed image array
    """
    if TF_AVAILABLE:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(32, 32))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
    else:
        try:
            from PIL import Image
            img = Image.open(image_path).resize((32, 32))
            img_array = np.array(img, dtype=np.float32)
        except ImportError:
            raise ImportError("Please install Pillow: pip install Pillow")
    
    # Normalize to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def predict_single(model: MicrocontrollerInference, image_array: np.ndarray) -> Dict[str, any]:
    """
    Perform inference on a single image.
    
    Args:
        model: MicrocontrollerInference instance
        image_array: Preprocessed image array
        
    Returns:
        Dictionary containing prediction results
    """
    # Run inference with timing
    predictions, inference_time = model.predict(image_array, measure_time=True)
    
    # Get predicted class and confidence
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    predicted_class = CLASS_NAMES[predicted_class_idx]
    
    # Get top 5 predictions
    top_5_idx = np.argsort(predictions[0])[-5:][::-1]
    top_5_predictions = [
        {
            'class': CLASS_NAMES[idx],
            'confidence': float(predictions[0][idx])
        }
        for idx in top_5_idx
    ]
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'inference_time_ms': inference_time,
        'top_5': top_5_predictions
    }


def batch_inference(model: MicrocontrollerInference, num_samples: int = 1000) -> Dict[str, any]:
    """
    Perform batch inference on CIFAR-10 test dataset.
    
    Args:
        model: MicrocontrollerInference instance
        num_samples: Number of samples to evaluate
        
    Returns:
        Dictionary containing batch inference results
    """
    print(f"\nLoading CIFAR-10 test dataset...")
    
    if not TF_AVAILABLE:
        print("Error: TensorFlow required for loading CIFAR-10 dataset")
        return {}
    
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype('float32') / 255.0
    y_test = y_test.squeeze()
    
    # Limit samples
    if num_samples > 0 and num_samples < len(x_test):
        x_test = x_test[:num_samples]
        y_test = y_test[:num_samples]
    
    print(f"Running INT8 inference on {len(x_test)} samples...")
    
    correct = 0
    start_time = time.time()
    
    # Process one by one
    for i, (image, label) in enumerate(zip(x_test, y_test)):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(x_test)} samples...")
        
        # Add batch dimension
        image_batch = np.expand_dims(image, axis=0)
        
        # Predict
        predictions, _ = model.predict(image_batch, measure_time=False)
        predicted_class = np.argmax(predictions[0])
        
        if predicted_class == label:
            correct += 1
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = correct / len(x_test)
    throughput = len(x_test) / total_time
    avg_latency = (total_time / len(x_test)) * 1000  # ms per image
    
    return {
        'total_samples': len(x_test),
        'accuracy': float(accuracy),
        'total_time_seconds': total_time,
        'throughput_images_per_sec': throughput,
        'avg_latency_ms_per_image': avg_latency
    }


def benchmark_latency(model: MicrocontrollerInference, num_runs: int = 100) -> Dict[str, any]:
    """
    Benchmark model inference latency.
    
    Args:
        model: MicrocontrollerInference instance
        num_runs: Number of inference runs
        
    Returns:
        Dictionary with latency statistics
    """
    print(f"\nRunning latency benchmark with {num_runs} inferences...")
    
    # Generate random input
    dummy_input = np.random.uniform(0, 1, size=(1, 32, 32, 3)).astype(np.float32)
    
    latencies = []
    
    # Warmup
    for _ in range(10):
        model.predict(dummy_input, measure_time=False)
    
    # Benchmark
    for i in range(num_runs):
        _, latency = model.predict(dummy_input, measure_time=True)
        latencies.append(latency)
        
        if (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{num_runs} runs...")
    
    latencies = np.array(latencies)
    
    return {
        'num_runs': num_runs,
        'mean_latency_ms': float(np.mean(latencies)),
        'median_latency_ms': float(np.median(latencies)),
        'min_latency_ms': float(np.min(latencies)),
        'max_latency_ms': float(np.max(latencies)),
        'std_latency_ms': float(np.std(latencies)),
        'p95_latency_ms': float(np.percentile(latencies, 95)),
        'p99_latency_ms': float(np.percentile(latencies, 99))
    }


def main():
    parser = argparse.ArgumentParser(description='Microcontroller Inference Script')
    parser.add_argument(
        '--model', 
        type=str, 
        default='models/microcontroller_int8.tflite',
        help='Path to the INT8 quantized TFLite model'
    )
    parser.add_argument(
        '--image', 
        type=str,
        help='Path to input image for single inference'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Run batch inference on test dataset'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run latency benchmark'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        help='Number of samples for batch inference'
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=100,
        help='Number of runs for benchmark'
    )
    
    args = parser.parse_args()
    
    # Load model
    model = MicrocontrollerInference(args.model)
    
    # Display memory info
    memory_info = model.get_memory_info()
    print("\n" + "="*60)
    print("MEMORY USAGE INFORMATION")
    print("="*60)
    print(f"Model Size:                 {memory_info['model_size_kb']:.2f} KB")
    print(f"Estimated Activation Mem:   {memory_info['estimated_activation_memory_kb']:.2f} KB")
    print(f"Total Estimated Memory:     {memory_info['total_estimated_memory_kb']:.2f} KB")
    print("="*60)
    
    if args.benchmark:
        # Benchmark mode
        print("\n" + "="*60)
        print("LATENCY BENCHMARK MODE")
        print("="*60)
        
        results = benchmark_latency(model, args.num_runs)
        
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        print(f"Number of Runs:      {results['num_runs']}")
        print(f"Mean Latency:        {results['mean_latency_ms']:.3f} ms")
        print(f"Median Latency:      {results['median_latency_ms']:.3f} ms")
        print(f"Min Latency:         {results['min_latency_ms']:.3f} ms")
        print(f"Max Latency:         {results['max_latency_ms']:.3f} ms")
        print(f"Std Deviation:       {results['std_latency_ms']:.3f} ms")
        print(f"P95 Latency:         {results['p95_latency_ms']:.3f} ms")
        print(f"P99 Latency:         {results['p99_latency_ms']:.3f} ms")
        print("="*60)
        
    elif args.batch:
        # Batch inference mode
        print("\n" + "="*60)
        print("BATCH INFERENCE MODE")
        print("="*60)
        
        results = batch_inference(model, args.num_samples)
        
        if results:
            print("\n" + "="*60)
            print("BATCH INFERENCE RESULTS")
            print("="*60)
            print(f"Total Samples:     {results['total_samples']}")
            print(f"Accuracy:          {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
            print(f"Total Time:        {results['total_time_seconds']:.2f} seconds")
            print(f"Throughput:        {results['throughput_images_per_sec']:.2f} images/sec")
            print(f"Avg Latency:       {results['avg_latency_ms_per_image']:.2f} ms/image")
            print("="*60)
        
    elif args.image:
        # Single image inference mode
        print("\n" + "="*60)
        print("SINGLE IMAGE INFERENCE MODE")
        print("="*60)
        
        # Preprocess image
        image_array = preprocess_image(args.image)
        
        # Perform inference
        results = predict_single(model, image_array)
        
        print("\n" + "="*60)
        print("INFERENCE RESULTS")
        print("="*60)
        print(f"Predicted Class:   {results['predicted_class']}")
        print(f"Confidence:        {results['confidence']:.4f} ({results['confidence']*100:.2f}%)")
        print(f"Inference Time:    {results['inference_time_ms']:.3f} ms")
        print("\nTop 5 Predictions:")
        for i, pred in enumerate(results['top_5'], 1):
            print(f"  {i}. {pred['class']:12s} - {pred['confidence']:.4f} ({pred['confidence']*100:.2f}%)")
        print("="*60)
        
    else:
        print("\nError: Please specify --image, --batch, or --benchmark mode")
        parser.print_help()
        return
    
    print("\nInference completed successfully!")
    print("\nNote: For actual microcontroller deployment, convert this model to")
    print("      TensorFlow Lite Micro (TFLM) format and deploy using C++.")


if __name__ == "__main__":
    main()
