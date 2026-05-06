import numpy as np

class LinearRegression(object):
    """Linear Regression model for regression tasks.
    
    Parameters
    ----------
    learning_rate: float, default=0.01
    The step size for updating the weights during gradient descent.
    n_iterations: int, default=1000
    The number of iterations for gradient descent.

    Attributes
    ----------
    w_: array of shape (n_features + 1,)
    The weights for the linear regression model."""

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.w_ = None

    def train_normal(self, X, y):
        """
        Train the linear regression model using the normal equation.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.w_ = np.linalg.lstsq(X_b, y, rcond=None)[0]

    def train_gd(self, X, y):
        """
        Train the linear regression model using gradient descent.

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

        for _ in range(self.n_iterations):
            errors = 0
            for xi, yi in zip(X, y):
                update = self.predict(xi) - yi
                self.w_[:-1] -= self.learning_rate * update * xi
                self.w_[-1] -= self.learning_rate * update
                errors += 0.5 * update ** 2
            self.errors_.append(errors / X.shape[0])
        return self

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
        linear_model = np.dot(X, self.w_[:-1]) + self.w_[-1]
        return linear_model