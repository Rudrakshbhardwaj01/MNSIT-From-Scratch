# MNSIT-From-Scratch

This repository implements the MNIST digit classification task **from scratch** using Python. The goal is to help learners understand the basics of machine learning and neural networks by building a simple classifier without relying on high-level deep learning libraries like TensorFlow, PyTorch, or Keras. The core logic is implemented manually, with the help of NumPy for numerical operations.

---

## Table of Contents

- [About](#about)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training & Evaluation](#training--evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## About

**MNIST From Scratch** is an educational project for understanding neural networks by building a digit classifier for the MNIST dataset, step by step. Here, you will find:

- Manual data loading and preprocessing (using NumPy)
- Building a neural network from scratch
- Forward and backward propagation
- Training loop with gradient descent
- Performance evaluation

---

## Features

- **Pure Python + NumPy:** No external ML frameworks required
- **Jupyter Notebook Implementation:** All logic and explanations are contained in a notebook for easy learning and experimentation
- **Manual Model Construction:** Forward and backward passes coded manually
- **Configurable Architecture:** Easily modify the network structure and hyperparameters

---

## Repository Structure

```
├── MNSITfromScratch.ipynb    # Main Jupyter Notebook: implementation and explanations
├── MNSITfromScratch/         # Directory (may contain additional resources or code)
├── README.md                 # Project documentation
```

All implementation and experimentation is intended to be done within `MNSITfromScratch.ipynb`.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rudrakshbhardwaj01/MNSIT-From-Scratch.git
   cd MNSIT-From-Scratch
   ```

2. **(Optional) Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   - Only NumPy and Jupyter are required; install with:
     ```bash
     pip install numpy jupyter
     ```

---

## Usage

1. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook MNSITfromScratch.ipynb
   ```

2. **Run the Notebook:**
   - Follow the steps in the notebook: load data, build the model, train, and evaluate.
   - Modify code cells to experiment with architecture, learning rate, epochs, etc.

---

## Dataset

- **Source:** [MNIST Database of handwritten digits](http://yann.lecun.com/exdb/mnist/)
- **Description:** 70,000 grayscale images (28x28 pixels), with 60,000 for training and 10,000 for testing.
- **Classes:** Digits 0–9

---

## Model Architecture

- **Input Layer:** 784 neurons (28x28 pixels flattened)
- **Hidden Layers:** Configurable; typical default is one layer with 128 neurons and ReLU activation
- **Output Layer:** 10 neurons (softmax for digit classification)

You can adjust the structure in the notebook code cells.

---

## Training & Evaluation

- **Forward Pass:** Computes predictions
- **Loss Function:** Cross-Entropy Loss
- **Backward Pass:** Manual gradient calculation
- **Optimizer:** Stochastic Gradient Descent (SGD)
- **Epochs and Batch Size:** Configurable in the notebook

Training progress and accuracy are displayed in output cells.

---

## Results

| Metric        | Value (Example) |
|---------------|----------------|
| Test Accuracy | ~91%           |

*Results depend on your hyperparameters and model structure.*

---

## Contributing

Contributions are welcome! Open issues or pull requests for:

- Bug fixes
- Feature additions
- Documentation clarifications
- Educational materials

---

## License

No explicit license file is present. For reuse or redistribution, please contact the repository owner.

---

## References

- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)

---

*For questions or suggestions, open an issue or contact [@Rudrakshbhardwaj01](https://github.com/Rudrakshbhardwaj01).*
