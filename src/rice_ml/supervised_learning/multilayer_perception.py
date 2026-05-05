import numpy as np
def sigmoid(z):
    """
    The sigmoid activation function.
    Squashes any real number into the range (0, 1).

        sigma(z) = 1 / (1 + exp(-z))
    """
    return 1.0 / (1.0 + np.exp(-z))

class MLP(object):
    def __init__(self, layer_sizes, activation="sigmoid"):
        self.activation = activation
        self.layer_sizes = layer_sizes
        self.w_ = []
        self.b_ = []

        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i-1]) * np.sqrt(2. / layer_sizes[i-1])
            b = np.random.randn(layer_sizes[i], 1) * np.sqrt(2. / layer_sizes[i-1])
            self.w_.append(w)
            self.b_.append(b)
 
    def forward(self, X):
        Z = []
        A = [X]
        L = len(w_)-1
        for i in range(1, L+1):
            z = W[i] @ A[i - 1] + B[i] 
        Z.append(z)
        a = sigmoid(z)                 
        A.append(a)
        return X
    
    def train(self, X, y, learning_rate=0.01, n_iterations=1000):
        self.errors_ = []
        for _ in range(n_iterations):
            pass
            for xi, yi in zip(X,y):
                pass
        return self.errors_
            