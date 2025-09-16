# MNSIT-From-Scratch

This repository implements the MNIST digit classification task **from scratch** using Python and NumPy. The aim is to offer an educational, step-by-step approach to understanding neural networks by manually building, training, and evaluating a simple classification model—without relying on frameworks like TensorFlow or PyTorch.

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

**MNIST From Scratch** is an educational project for learning about neural networks by constructing a digit classifier for the MNIST dataset from the ground up. This repository guides you through:

- Manual data loading and preprocessing (using NumPy)
- Building a neural network from scratch with multiple layers
- Implementing forward and backward propagation
- Training the model using gradient descent
- Evaluating model performance

---

## Features

- **Pure Python + NumPy:** No external ML frameworks required.
- **Jupyter Notebook Implementation:** All logic and explanations are contained in a notebook for easy learning and experimentation.
- **Manual Model Construction:** Forward and backward passes coded by hand.
- **Configurable Architecture:** Easily modify the network structure and hyperparameters.

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
- **Hidden Layers:** 3 hidden layers (512, 128, 64 neurons respectively with ReLU activation)
- **Output Layer:** 10 neurons (softmax for digit classification)

The structure is easily adjustable in the notebook.

---

## Training & Evaluation

- **Forward Pass:** Computes predictions through the network.
- **Loss Function:** Cross-Entropy Loss.
- **Backward Pass:** Manual calculation of gradients via backpropagation.
- **Optimizer:** Stochastic Gradient Descent (SGD).
- **Epochs and Batch Size:** Configurable in the notebook.

During training, progress and accuracy are displayed at intervals.

---

## Results

| Metric        | Value (Observed) |
|---------------|------------------|
| Train Accuracy | up to ~99.97%   |
| Dev/Test Accuracy | ~95.8%       |

*Final accuracy may vary slightly based on random initialization and hyperparameters. The model achieves high training accuracy and strong generalization on the dev set using a fully manual, multilayer neural network.*

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
