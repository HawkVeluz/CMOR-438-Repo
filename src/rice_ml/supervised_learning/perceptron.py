import numpy as np

class Perceptron:
    """ Perceptron classifier for binary classification tasks.
    Parameters
    ----------
    learning_rate : float, default=0.01
        The learning rate for the weight updates during training.
    n_iterations : int, default=1000
        The number of iterations for training the perceptron.
    
    Attributes
    ----------
    w_ : ndarray of shape (n_features + 1,)
        The learned weights after training, including the bias term as the last element.
    errors_ : list
        The list of misclassifications for each iteration during training.
    """
    def __init__(self, learning_rate=0.1, n_iters=1000):
        self.eta = learning_rate
        self.n_iters = n_iters
    
    def train(self, X, y):
        """
        Train the perceptron model on the provided data.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        self.w_ = np.random.rand(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iters):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (self.predict(xi) - target)
                self.w_[:-1] -= update * xi
                self.w_[-1] -= update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[:-1]) + self.w_[-1]
    
    def predict(self, X):
        """Predict the target values for the provided data.
        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        
        Attributes
        -------
        y_pred: ndarray of shape (n_samples,)
        Predicted target values for each data point in X.
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)