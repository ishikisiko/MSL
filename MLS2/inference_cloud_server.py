"""
Cloud Server Inference Script
================================
This script demonstrates how to perform inference using the optimized cloud model.
Suitable for high-performance cloud environments with abundant resources.

Requirements:
- TensorFlow 2.x
- NumPy
- Sufficient RAM (>512MB recommended)

Usage:
    python inference_cloud_server.py --model models/cloud_server_optimized.keras --image path/to/image.png
    python inference_cloud_server.py --batch  # Run batch inference on test dataset
"""

import tensorflow as tf
import numpy as np
import argparse
import os
import time
from typing import Tuple, List, Dict

# CIFAR-10 class names
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def load_model(model_path: str) -> tf.keras.Model:
    """
    Load the optimized Keras model from disk.
    
    Args:
        model_path: Path to the saved Keras model
        
    Returns:
        Loaded TensorFlow/Keras model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded successfully!")
    print(f"Model summary:")
    model.summary()
    
    return model


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess a single image for inference.
    
    Args:
        image_path: Path to input image
        
    Returns:
        Preprocessed image array ready for inference
    """
    # Load image
    img = tf.keras.preprocessing.image.load_img(
        image_path, 
        target_size=(32, 32)  # CIFAR-10 image size
    )
    
    # Convert to array and normalize
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def predict_single(model: tf.keras.Model, image_array: np.ndarray) -> Dict[str, any]:
    """
    Perform inference on a single image.
    
    Args:
        model: Loaded Keras model
        image_array: Preprocessed image array
        
    Returns:
        Dictionary containing prediction results
    """
    start_time = time.time()
    
    # Perform inference
    predictions = model.predict(image_array, verbose=0)
    
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


def batch_inference(model: tf.keras.Model, batch_size: int = 32, num_samples: int = 1000) -> Dict[str, any]:
    """
    Perform batch inference on CIFAR-10 test dataset.
    
    Args:
        model: Loaded Keras model
        batch_size: Batch size for inference
        num_samples: Number of samples to evaluate
        
    Returns:
        Dictionary containing batch inference results
    """
    print(f"\nLoading CIFAR-10 test dataset...")
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype('float32') / 255.0
    y_test = y_test.squeeze()
    
    # Limit samples
    if num_samples > 0 and num_samples < len(x_test):
        x_test = x_test[:num_samples]
        y_test = y_test[:num_samples]
    
    print(f"Running batch inference on {len(x_test)} samples (batch_size={batch_size})...")
    
    start_time = time.time()
    predictions = model.predict(x_test, batch_size=batch_size, verbose=1)
    total_time = time.time() - start_time
    
    # Calculate accuracy
    predicted_classes = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_classes == y_test)
    
    # Calculate throughput
    throughput = len(x_test) / total_time
    avg_latency = (total_time / len(x_test)) * 1000  # ms per image
    
    return {
        'total_samples': len(x_test),
        'batch_size': batch_size,
        'accuracy': float(accuracy),
        'total_time_seconds': total_time,
        'throughput_images_per_sec': throughput,
        'avg_latency_ms_per_image': avg_latency
    }


def main():
    parser = argparse.ArgumentParser(description='Cloud Server Inference Script')
    parser.add_argument(
        '--model', 
        type=str, 
        default='models/cloud_server_optimized.keras',
        help='Path to the optimized Keras model'
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
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for batch inference'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        help='Number of samples for batch inference'
    )
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    
    if args.batch:
        # Batch inference mode
        print("\n" + "="*60)
        print("BATCH INFERENCE MODE")
        print("="*60)
        
        results = batch_inference(model, args.batch_size, args.num_samples)
        
        print("\n" + "="*60)
        print("BATCH INFERENCE RESULTS")
        print("="*60)
        print(f"Total Samples:     {results['total_samples']}")
        print(f"Batch Size:        {results['batch_size']}")
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
