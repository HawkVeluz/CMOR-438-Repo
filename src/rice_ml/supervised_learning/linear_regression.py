import numpy as np

class LinearRegression(object):
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.w_ = None

    def train_normal(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.w_ = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

    def train_gd(self, X, y):
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
        linear_model = np.dot(X, self.w_[:-1]) + self.w_[-1]
        return linear_model