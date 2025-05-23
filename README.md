# Digit Recognizer Using Neural Network (From Scratch)

This project implements a neural network from scratch to recognize handwritten digits from the **MNIST dataset**. It avoids using pre-built libraries for the neural network, focusing on understanding the fundamental principles of machine learning.

## Features
- Manual implementation of forward propagation, backpropagation, and gradient descent.
- Trains a feedforward neural network for classifying digits (0-9).
- Uses the MNIST dataset for training and testing.

## Dataset
The MNIST dataset consists of:
- 60,000 training images.
- 10,000 testing images.
Each image is a 28x28 grayscale picture of a digit.

## Key Components
- **Input Layer**: 784 nodes (28x28 pixels flattened).
- **Hidden Layers**: Customizable number of neurons and activation functions.
- **Output Layer**: 10 nodes (one for each digit).
- **Activation Function**: ReLU and softmax.

## How to Run
1. Clone this repository.
   ```bash
   git clone https://github.com/your-username/digit-recognizer.git
   cd digit-recognizer

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
The network is implemented in the neural_net.py file. 
Key components include:
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

Results
---

Achieved an accuracy of ~95% on the MNIST test dataset.
Training and evaluation time depends on hardware but is typically efficient on most modern CPUs.

Future Work
---

Experiment with different architectures (e.g., adding more hidden layers or neurons).
Implement advanced optimization techniques (e.g., Adam or RMSprop).
Extend to other datasets like CIFAR-10 or Fashion MNIST.

Contributing
---
Contributions are welcome! If you find any bugs or want to improve the project, feel free to open an issue or submit a pull request.

License
---
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
---
MNIST Database by Yann LeCun for the dataset.
Inspiration from fundamental machine learning courses and books.
