import numpy as np
import pytest
from rice_ml.supervised_learning import LogisticRegression

# Logistic regression binary output
def test_logistic_regression_binary_output():
    X = np.array([[0], [1], [2], [3], [4]])
    y = np.array([0, 0, 0, 1, 1])

    model = LogisticRegression()
    model.train(X, y)

    pred = model.predict(X)

    assert set(np.unique(pred)).issubset({0, 1})


# Logistic regression shape
def test_logistic_regression_shape():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression()
    model.train(X, y)

    pred = model.predict(X)

    assert pred.shape == y.shape


# Logistic regression no NaN
def test_logistic_regression_no_nan():
    X = np.random.rand(40, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    model = LogisticRegression()
    model.train(X, y)

    pred = model.predict(X)

    assert np.all(np.isfinite(pred))


# Logistic regression separable data
def test_logistic_regression_separable_data():
    X = np.array([[0], [1], [2], [3], [4], [5]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticRegression(learning_rate=0.1, n_iterations=200)
    model.train(X, y)

    pred = model.predict(X)

    assert np.mean(pred == y) >= 0.8
