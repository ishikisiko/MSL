import tensorflow as tf
import os
import numpy as np
from part1_tensorflow import load_and_preprocess_data


def convert_to_tflite(model_path, quantize=False):
    """
    Convert TensorFlow model to TensorFlow Lite format.

    Args:
        model_path (str): Path to saved TensorFlow model
        quantize (bool): Whether to apply quantization

    Returns:
        bytes: TensorFlow Lite model data
    """
    # Load the saved model
    model = tf.keras.models.load_model(model_path)
    
    # Create TensorFlow Lite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply quantization if requested
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
    # Convert model and return tflite data
    tflite_model_data = converter.convert()
    return tflite_model_data


def analyze_model_size(tf_model_path, tflite_model_data):
    """
    Compare model sizes between TensorFlow and TensorFlow Lite.

    Args:
        tf_model_path (str): Path to TensorFlow model
        tflite_model_data (bytes): TensorFlow Lite model data
    """
    # Calculate file sizes and compression ratio
    tf_model_size = os.path.getsize(tf_model_path)
    tflite_model_size = len(tflite_model_data)
    compression_ratio = tf_model_size / tflite_model_size
    
    # Print comparison results
    print(f"TensorFlow model size: {tf_model_size / 1024:.2f} KB")
    print(f"TensorFlow Lite model size: {tflite_model_size / 1024:.2f} KB")
    print(f"Compression ratio: {compression_ratio:.2f}x")


def test_tflite_model(tflite_model_data, x_test, y_test):
    """
    Test TensorFlow Lite model accuracy and loss.

    Args:
        tflite_model_data (bytes): TensorFlow Lite model
        x_test, y_test: Test data

    Returns:
        tuple: (Test loss, Test accuracy)
    """
    # Create TensorFlow Lite interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model_data)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    correct_predictions = 0
    total_loss = 0.0
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Run inference on test data
    for i in range(len(x_test)):
        # Preprocess the input image to match the model's input shape
        img = np.expand_dims(x_test[i], axis=0).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_label = np.argmax(output_data)
        
        # Calculate loss for the current sample
        loss = loss_fn(y_test[i:i+1], output_data).numpy()
        total_loss += loss
        
        if predicted_label == y_test[i]:
            correct_predictions += 1
            
    # Calculate and return average loss and accuracy
    accuracy = correct_predictions / len(x_test)
    average_loss = total_loss / len(x_test)
    return average_loss, accuracy

if __name__ == "__main__":
    # Convert model without quantization
    tflite_model = convert_to_tflite('mnist_cnn_model.keras', quantize=False)
    
    # Convert model with quantization
    tflite_quantized_model = convert_to_tflite('mnist_cnn_model.keras', quantize=True)
    
    # Save TensorFlow Lite models
    with open('mnist_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    with open('mnist_model_quantized.tflite', 'wb') as f:
        f.write(tflite_quantized_model)
    
    # Analyze model sizes
    analyze_model_size('mnist_cnn_model.keras', tflite_model)
    analyze_model_size('mnist_cnn_model.keras', tflite_quantized_model)
    
    # Test accuracy of converted models
    # (You'll need to load test data again)
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    
    tflite_loss, tflite_accuracy = test_tflite_model(tflite_model, x_test, y_test)
    tflite_quantized_loss, tflite_quantized_accuracy = test_tflite_model(tflite_quantized_model, x_test, y_test)
    
    print(f"TensorFlow Lite loss: {tflite_loss:.4f}, accuracy: {tflite_accuracy:.4f}")
    print(f"TensorFlow Lite quantized loss: {tflite_quantized_loss:.4f}, accuracy: {tflite_quantized_accuracy:.4f}")