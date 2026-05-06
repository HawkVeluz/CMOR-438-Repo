# Multilayer Perceptron (MLP)

The Multilayer Perceptron is a feedforward neural network that learns nonlinear relationships using layered linear transformations and activation functions. It is trained using backpropagation and gradient descent.


## Algorithm

1. Initialize weights and biases for each layer

2. Forward pass:
   - Compute weighted sums
   - Apply activation function (sigmoid)

3. Compute loss (e.g., MSE or cross-entropy)

4. Backward pass:
   - Compute gradients using chain rule
   - Propagate errors backward through layers

5. Update weights using gradient descent

6. Repeat for multiple epochs


## UCI Dataset


## Files

- `MLP.ipynb` — implementation and training on classification dataset
