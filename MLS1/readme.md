Python (run in windows)
===================================================
Part 1: TensorFlow Model Training
requires: tensorflow
runs: python part1_tensorflow.py


Part 2: Model Conversion to TensorFlow Lite
runs: python part2_tflite_conversion.py
===================================================
C++ (run in linux)
requires: TensorFlow Lite micro
mnist_5_samples.h: contains 5 sample MNIST images for quick testing
Part 3: TensorFlow Lite Model Inference
runs: make
      ./mnist_inference