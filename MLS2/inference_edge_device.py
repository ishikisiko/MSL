"""
Edge Device Inference Script
==============================
This script demonstrates how to perform inference using the optimized edge model (TFLite).
Suitable for edge devices like Raspberry Pi, NVIDIA Jetson, or mobile devices.

Requirements:
- TensorFlow Lite Runtime (or full TensorFlow)
- NumPy
- Limited RAM (>128MB recommended)

Usage:
    python inference_edge_device.py --model models/edge_device_dynamic.tflite --image path/to/image.png
    python inference_edge_device.py --batch  # Run batch inference on test dataset
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


class TFLiteInference:
    """
    Wrapper class for TFLite inference with support for both quantized and float models.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize TFLite interpreter.
        
        Args:
            model_path: Path to the TFLite model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        print(f"Loading TFLite model from: {model_path}")
        
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
        
        # Print model information
        self._print_model_info()
    
    def _print_model_info(self):
        """Print model input/output information."""
        print(f"\nModel loaded successfully!")
        print(f"\nInput details:")
        for detail in self.input_details:
            print(f"  Name: {detail['name']}")
            print(f"  Shape: {detail['shape']}")
            print(f"  Type: {detail['dtype']}")
            if detail['quantization'] != (0.0, 0):
                scale, zero_point = detail['quantization']
                print(f"  Quantization: scale={scale}, zero_point={zero_point}")
        
        print(f"\nOutput details:")
        for detail in self.output_details:
            print(f"  Name: {detail['name']}")
            print(f"  Shape: {detail['shape']}")
            print(f"  Type: {detail['dtype']}")
            if detail['quantization'] != (0.0, 0):
                scale, zero_point = detail['quantization']
                print(f"  Quantization: scale={scale}, zero_point={zero_point}")
    
    def preprocess_input(self, image_array: np.ndarray) -> np.ndarray:
        """
        Preprocess input based on model's input type.
        
        Args:
            image_array: Input image array (float32, normalized to [0, 1])
            
        Returns:
            Preprocessed array matching model's input requirements
        """
        input_dtype = self.input_details[0]['dtype']
        
        if input_dtype == np.int8:
            # Quantized model - convert float to int8
            input_scale, input_zero_point = self.input_details[0]['quantization']
            
            if input_scale > 0:
                # Use quantization parameters
                quantized = (image_array / input_scale + input_zero_point).astype(np.int8)
            else:
                # Fallback: map [0, 1] to [-128, 127]
                quantized = ((image_array - 0.5) * 255).astype(np.int8)
            
            return quantized
        else:
            # Float model - use as is
            return image_array.astype(np.float32)
    
    def postprocess_output(self, output_data: np.ndarray) -> np.ndarray:
        """
        Postprocess output based on model's output type.
        
        Args:
            output_data: Raw output from model
            
        Returns:
            Dequantized output (float32)
        """
        output_dtype = self.output_details[0]['dtype']
        
        if output_dtype == np.int8:
            # Dequantize output
            output_scale, output_zero_point = self.output_details[0]['quantization']
            
            if output_scale > 0:
                dequantized = (output_data.astype(np.float32) - output_zero_point) * output_scale
            else:
                # Fallback
                dequantized = output_data.astype(np.float32) / 127.0
            
            return dequantized
        else:
            # Already float
            return output_data
    
    def predict(self, image_array: np.ndarray) -> np.ndarray:
        """
        Run inference on input image.
        
        Args:
            image_array: Input image (shape: [1, 32, 32, 3], normalized to [0, 1])
            
        Returns:
            Model predictions (shape: [1, num_classes])
        """
        # Preprocess input
        input_data = self.preprocess_input(image_array)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output tensor
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Postprocess output
        output_data = self.postprocess_output(output_data)
        
        return output_data


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess a single image for inference.
    
    Args:
        image_path: Path to input image
        
    Returns:
        Preprocessed image array
    """
    if TF_AVAILABLE:
        # Use TensorFlow's image utilities
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(32, 32))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
    else:
        # Fallback: use PIL or other image library
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


def predict_single(model: TFLiteInference, image_array: np.ndarray) -> Dict[str, any]:
    """
    Perform inference on a single image.
    
    Args:
        model: TFLiteInference instance
        image_array: Preprocessed image array
        
    Returns:
        Dictionary containing prediction results
    """
    start_time = time.time()
    
    # Perform inference
    predictions = model.predict(image_array)
    
    inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
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


def batch_inference(model: TFLiteInference, num_samples: int = 1000) -> Dict[str, any]:
    """
    Perform batch inference on CIFAR-10 test dataset.
    
    Args:
        model: TFLiteInference instance
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
    
    print(f"Running inference on {len(x_test)} samples...")
    
    correct = 0
    start_time = time.time()
    
    # Process one by one (TFLite inference is typically single-sample)
    for i, (image, label) in enumerate(zip(x_test, y_test)):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(x_test)} samples...")
        
        # Add batch dimension
        image_batch = np.expand_dims(image, axis=0)
        
        # Predict
        predictions = model.predict(image_batch)
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


def main():
    parser = argparse.ArgumentParser(description='Edge Device Inference Script')
    parser.add_argument(
        '--model', 
        type=str, 
        default='models/edge_device_dynamic.tflite',
        help='Path to the TFLite model'
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
        '--num-samples',
        type=int,
        default=1000,
        help='Number of samples for batch inference'
    )
    
    args = parser.parse_args()
    
    # Load model
    model = TFLiteInference(args.model)
    
    if args.batch:
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
        print(f"Inference Time:    {results['inference_time_ms']:.2f} ms")
        print("\nTop 5 Predictions:")
        for i, pred in enumerate(results['top_5'], 1):
            print(f"  {i}. {pred['class']:12s} - {pred['confidence']:.4f} ({pred['confidence']*100:.2f}%)")
        print("="*60)
        
    else:
        print("\nError: Please specify either --image or --batch mode")
        parser.print_help()
        return
    
    print("\nInference completed successfully!")


if __name__ == "__main__":
    main()
