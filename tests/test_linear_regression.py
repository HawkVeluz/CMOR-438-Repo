import numpy as np
import pytest
from rice_ml.supervised_learning import LinearRegression

# Linear regression shapes
def test_linear_regression_shapes():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])

    model = LinearRegression()
    model.train_gd(X, y)

    pred = model.predict(X)

    assert pred.shape == y.shape


# Linear regression perfect train_gd
def test_linear_regression_perfect_train_gd():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])

    model = LinearRegression()
    model.train_gd(X, y)

    pred = model.predict(X)

    assert np.allclose(pred, y, atol=1e-1)


# Linear regression no NaN
def test_linear_regression_no_nan():
    X = np.random.rand(50, 3)
    y = np.random.rand(50)

    model = LinearRegression()
    model.train_gd(X, y)

    pred = model.predict(X)

    assert np.all(np.isfinite(pred))


# Linear regression loss decreases
def test_linear_regression_loss_decreases():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])

    model = LinearRegression()
    model.train_gd(X, y)

    assert hasattr(model, "errors_")
    assert model.errors_[-1] <= model.errors_[0]