import numpy as np

def sigmoid(z):
    """ 
    The sigmoid activation function.
    Squashes any real number into the range (0, 1).
        sigma(z) = 1 / (1 + exp(-z))
    """
    return 1 / (1 + np.exp(-z))

class LogisticRegression(object):
    """ Logistic Regression classifier using Stochastic Gradient Descent. 
    Parameters
    ----------
    learning_rate : float, default=0.01
        The learning rate for the gradient descent algorithm.
    n_iterations : int, default=1000
        The number of iterations for the gradient descent algorithm.
    
    Attributes
    ----------
    w_ : ndarray of shape (n_features + 1,)
        The learned weights after training, including the bias term as the last element.
    errors_ : list
        The list of average cross-entropy loss values for each iteration during training.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations= n_iterations

    def predict_prob(self, X):
        """
        Predict the probabilities for the provided data.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        y_prob: ndarray of shape (n_samples,)
            The predicted probabilities for each data point in X.
        """
        # Calculate the predicted probabilities for samples in X.
        z = np.dot(X, self.w_[:-1]) + self.w_[-1]
        return sigmoid(z)

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
        # Predict class labels for samples in X.
        return (self.predict_prob(X) >= 0.5).astype(int)

    def train(self, X, y):
        """
        Train the logistic regression model on the provided data.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        self.w_ = np.random.rand(X.shape[1] + 1)
        self.errors_ = []

        for _ in range(self.n_iterations):
            errors = 0
            for xi, yi in zip(X, y):
                p = sigmoid(np.dot(xi, self.w_[:-1]) + self.w_[-1])
                update = p - yi
                self.w_[:-1] -= self.learning_rate * update * xi
                self.w_[-1] -= self.learning_rate * update
                # Compute cross-entropy loss for this sample and accumulate
                errors += -yi * np.log(p + 1e-10) - (1 - yi) * np.log(1 - p + 1e-10)
            self.errors_.append(errors / X.shape[0])
        return self