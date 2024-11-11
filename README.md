Digit Recognizer Using Neural Network (From Scratch)
---

Overview
---

This project implements a neural network from scratch using Python to recognize handwritten digits from the popular MNIST dataset. The goal is to demonstrate a fundamental understanding of neural networks by manually coding forward propagation, backpropagation, and gradient descent without using high-level machine learning libraries like TensorFlow or PyTorch.

Features
---

Handwritten digit recognition using a feedforward neural network.
Complete implementation of:
Weight initialization.
Forward propagation.
Backpropagation.
Activation functions (ReLU, Sigmoid, Softmax).
Gradient descent optimization.
Visualization of the loss curve during training.
Evaluation on unseen test data.

Dataset
---
MNIST Dataset: Contains 60,000 training images and 10,000 test images of handwritten digits (0-9), each represented as a 28x28 grayscale image.
The dataset can be downloaded from MNIST Database.

Key Components
---

1. Neural Network Implementation
---


The network is implemented in the neural_net.py file. Key components include:

Layers: Input, hidden, and output layers.
Activation Functions:
ReLU: For hidden layers.
Softmax: For the output layer.
Loss Function: Categorical Cross-Entropy.
Optimization: Gradient Descent.

2. Data Preprocessing
---

Normalization: Scale pixel values to the range [0, 1].
One-hot Encoding: Transform labels into one-hot vectors for classification.
