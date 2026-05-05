import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression(object):

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations= n_iterations

    def predict_prob(self, X):
        z = np.dot(X, self.w_[:-1]) + self.w_[-1]
        return sigmoid(z)

    def predict(self, X):
        return (self.predict_prob(X) >= 0.5).astype(int)

    def train(self, X, y):
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        n_samples, n_features = X.shape
        self.w_ = np.random.rand(n_features + 1)
        self.errors_ = []

        for _ in range(self.n_iterations):
            errors = 0
            for xi, yi in zip(X, y):
                p = sigmoid(np.dot(xi, self.w_[:-1]) + self.w_[-1])
                update = p - yi
                self.w_[:-1] -= self.learning_rate * update * xi
                self.w_[-1] -= self.learning_rate * update
                errors += -yi * np.log(p + 1e-10) - (1 - yi) * np.log(1 - p + 1e-10)
            self.errors_.append(errors / X.shape[0])
        return self