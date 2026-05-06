import numpy as np
import pytest
from rice_ml.supervised_learning import Perceptron 

# AND gate (linearly separable)
def test_perceptron_and_gate():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 0, 0, 1])

    model = Perceptron(learning_rate=0.1, n_iters=20)
    model.train(X, y)
    pred = model.predict(X)

    assert np.array_equal(pred, y)


# OR gate
def test_perceptron_or_gate():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 1, 1, 1])

    model = Perceptron(learning_rate=0.1, n_iters=20)
    model.train(X, y)
    pred = model.predict(X)

    assert np.array_equal(pred, y)

# Shape check
def test_perceptron_output_shape():
    X = np.array([[1, 2], [2, 3], [3, 4]])
    y = np.array([0, 1, 1])

    model = Perceptron()
    model.train(X, y)

    pred = model.predict([[1, 2]])

    assert pred.shape == (1,)

# No NaN outputs

def test_perceptron_no_nan():
    X = np.random.rand(30, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    model = Perceptron()
    model.train(X, y)

    pred = model.predict(X)

    assert np.all(np.isfinite(pred))

# Error changes with iterations
def test_perceptron_learning_progress():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 0, 0, 1])

    model = Perceptron(learning_rate=0.1, n_iters=50)
    model.train(X, y)

    assert hasattr(model, "errors_")

    assert len(model.errors_) > 0
    assert model.errors_[-1] <= model.errors_[0]
