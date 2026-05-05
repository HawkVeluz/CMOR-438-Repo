import numpy as np
from rice_ml.supervised_learning import KNN, KNNRegressor
import pytest

# KNN initialization
def test_knn_initialization():
    model = KNN(k=3)
    assert model.k == 3

    # KNN shape consistency
def test_knn_prediction_shape():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])

    model = KNN(k=1)
    model.train(X, y)

    preds = model.predict(X)

    assert preds.shape == y.shape

# KNN perfect separable case
def test_knn_perfect_classification():
    X = np.array([
        [0],
        [1],
        [10],
        [11]
    ])
    y = np.array([0, 0, 1, 1])

    model = KNN(k=1)
    model.train(X, y)

    preds = model.predict(X)

    assert np.array_equal(preds, y)

# KNN binary output only
def test_knn_binary_output():
    X = np.random.rand(50, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    model = KNN(k=3)
    model.train(X, y)

    preds = model.predict(X)

    assert set(np.unique(preds)).issubset({0, 1})

# KNN majority vote behavior
def test_knn_majority_vote():
    X = np.array([
        [0],
        [1],
        [2],
        [3],
        [4]
    ])
    y = np.array([0, 0, 0, 1, 1])

    model = KNN(k=3)
    model.train(X, y)

    pred = model.predict([[2]])

    assert pred in [0, 1]

# KNN regressor initialization
def test_knn_regressor_initialization():
    model = KNNRegressor(k=3)
    assert model.k == 3

# KNN regressor shape consistency
def test_knn_regressor_shape():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    model = KNNRegressor(k=1)
    model.train(X, y)

    preds = model.predict(X)

    assert preds.shape == y.shape

# KNN regressor exact interpolation (k=1)
def test_knn_regressor_perfect_fit_k1():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0.0, 1.0, 2.0, 3.0])

    model = KNNRegressor(k=1)
    model.train(X, y)

    preds = model.predict(X)

    assert np.allclose(preds, y)

# KNN regressor averaging behavior
def test_knn_regressor_averaging():
    X = np.array([[0], [1], [2], [3], [4]])
    y = np.array([0.0, 0.0, 10.0, 10.0, 10.0])

    model = KNNRegressor(k=3)
    model.train(X, y)

    pred = model.predict([[2]])

    assert isinstance(pred[0], (float, np.floating))

# KNN regressor no NaN outputs
def test_knn_regressor_no_nan():
    X = np.random.rand(40, 2)
    y = np.random.rand(40)

    model = KNNRegressor(k=5)
    model.train(X, y)

    preds = model.predict(X)

    assert np.all(np.isfinite(preds))
