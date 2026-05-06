import numpy as np

def sigmoid(z):
    """
    Sigmoid activation function.

    sigmoid(z) = 1 / (1 + exp(-z))
    """
    z = np.clip(z, -500, 500)  # prevent overflow
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


class MLP(object):
    """
    Multilayer Perceptron (MLP) for binary classification.
    Parameters
    ----------
    layer_sizes : list of int
        List specifying the number of neurons in each layer (including input and output layers).
        activation : str, default="sigmoid"
        The activation function to use for the hidden layers. Currently only "sigmoid" is supported.
    Attributes
    ----------
    w_ : list of array-like
        Weights for each layer.
    b_ : list of array-like
        Biases for each layer.
    """

    def __init__(self, layer_sizes, activation="sigmoid"):
        self.activation = activation
        self.layer_sizes = layer_sizes

        self.w_ = []
        self.b_ = []

        # weight initialization
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * np.sqrt(2. / layer_sizes[i])
            b = np.zeros((layer_sizes[i + 1], 1))

            self.w_.append(w)
            self.b_.append(b)

    def forward(self, X):
        """
        Forward propagation through the network.
        """
        A = X.T
        As = [A]
        Zs = []

        L = len(self.w_)

        for i in range(L):
            Z = self.w_[i] @ A + self.b_[i]
            A = sigmoid(Z)

            Zs.append(Z)
            As.append(A)

        return As, Zs

    def backward(self, X, y, activations, zs, learning_rate):
        """
        Backpropagation to compute gradients.
        """
        m = X.shape[0]
        y = y.reshape(1, -1)

        # output layer error
        delta = (activations[-1] - y) * sigmoid_derivative(zs[-1])

        for i in reversed(range(len(self.w_))):
            dw = (delta @ activations[i].T) / m
            db = np.sum(delta, axis=1, keepdims=True) / m

            self.w_[i] -= learning_rate * dw
            self.b_[i] -= learning_rate * db

            if i > 0:
                delta = (self.w_[i].T @ delta) * sigmoid_derivative(zs[i - 1])

    def train(self, X, y, learning_rate=0.01, n_iterations=1000):
        """
        Train MLP using gradient descent.
        """
        X = np.array(X)
        y = np.array(y)
        self.errors_ = []

        for _ in range(n_iterations):

            activations, zs = self.forward(X)

            loss = np.mean((activations[-1] - y.reshape(1, -1)) ** 2)
            self.errors_.append(loss)

            self.backward(X, y, activations, zs, learning_rate)

        return self.errors_

    def predict(self, X):
        X = np.array(X)
        activations, _ = self.forward(X)
        return (activations[-1] > 0.5).astype(int).flatten()