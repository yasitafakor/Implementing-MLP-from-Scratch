# Implementing MLP from Scratch

## Project Overview

This project demonstrates the implementation of a Multi-Layer Perceptron (MLP) from scratch using Python. The focus is on building foundational components like a custom `Tensor` class, layers, and an optimizer, all while enabling automatic differentiation for training deep learning models.

## Tensor Class

The `Tensor` class is the backbone of this project, representing multi-dimensional arrays with support for automatic differentiation.

### Key Features:
- **Attributes**: `value`, `children`, `operator`, `grad`, `backward`.
- **Operations**: Implements basic operations (`__add__`, `__sub__`, `__mul__`, `__truediv__`, `__pow__`) with backward propagation support using topological sorting.

## Neuron Class

The `Neuron` class models a single neuron in a neural network.

### Key Features:
- **Attributes**: `weights`, `bias` (both initialized randomly).
- **Forward Method**: Computes the weighted sum of inputs, adds bias, and applies the `tanh` activation function.

## Layer Class

The `Layer` class represents a single layer in the neural network, consisting of multiple neurons.

### Key Features:
- **Attributes**: `neurons` (a list of `Neuron` objects).
- **Forward Method**: Passes inputs through each neuron in the layer.

## MLP Class

The `MLP` class represents the entire neural network, composed of multiple layers.

### Key Features:
- **Attributes**: `layers` (a list of `Layer` objects).
- **Forward Method**: Passes inputs through all layers sequentially.

## Optimizer Class

The `Optimizer` class updates the model’s parameters based on the computed gradients.

### Key Features:
- **Attributes**: `params` (model parameters), `lr` (learning rate).
- **Step Method**: Adjusts weights and biases by a factor of `lr * grad`.

## F Class (Activation Functions)

The `F` class provides activation functions like `tanh` and `sigmoid`, along with their backward methods.

### Key Features:
- **`tanh(x)`**: Squashes inputs between `-1` and `1`.
- **`sigmoid(x)`**: Maps inputs to a range between `0` and `1`.

## Criterion Function

The `criterion` function computes the Mean Squared Error (MSE) between predicted and actual values, serving as the loss function during training.

## Training Process

The training process involves defining the model architecture, setting up the optimizer, and iterating through epochs to minimize the loss.

1. **Model Initialization**: Define input, output, and hidden layer sizes.
2. **Optimizer Setup**: Create an optimizer with model parameters and learning rate.
3. **Training Loop**:
    - Forward pass to compute predictions.
    - Calculate loss using `criterion`.
    - Perform backward pass to compute gradients.
    - Update parameters with the optimizer.

