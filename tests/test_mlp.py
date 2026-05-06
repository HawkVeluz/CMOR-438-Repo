import numpy as np
import pytest
from rice_ml.supervised_learning import MLP


# test forward output shape
def test_forward_shape():
    X = np.random.rand(10, 4)
    model = MLP(layer_sizes=[4, 5, 1])

    activations, _ = model.forward(X)

    # output should be (1, n_samples)
    assert activations[-1].shape == (1, 10)


# test predict shape
def test_predict_shape():
    X = np.random.rand(8, 3)
    model = MLP(layer_sizes=[3, 4, 1])

    preds = model.predict(X)

    assert preds.shape == (8,)


# test outputs are binary
def test_binary_output():
    X = np.random.rand(6, 2)
    model = MLP(layer_sizes=[2, 3, 1])

    preds = model.predict(X)

    assert set(preds).issubset({0, 1})


# test loss decreases over training
def test_loss_decreases():
    # simple linearly separable dataset
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 0, 0, 1])

    model = MLP(layer_sizes=[2, 4, 1])
    errors = model.train(X, y, learning_rate=0.1, n_iterations=200)

    assert errors[-1] < errors[0]


# test learning XOR (nonlinear capability)
def test_xor_learning():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 1, 1, 0])

    model = MLP(layer_sizes=[2, 5, 1])
    model.train(X, y, learning_rate=0.1, n_iterations=2000)

    preds = model.predict(X)

    # should learn XOR reasonably well
    assert np.sum(preds == y) >= 3  # allow small error


# test weights update during training
def test_weights_update():
    X = np.random.rand(5, 3)
    y = np.array([0, 1, 0, 1, 0])

    model = MLP(layer_sizes=[3, 4, 1])

    initial_weights = [w.copy() for w in model.w_]

    model.train(X, y, n_iterations=10)

    for w_init, w_new in zip(initial_weights, model.w_):
        assert not np.allclose(w_init, w_new)


# test zero input
def test_zero_input():
    X = np.zeros((4, 3))
    y = np.array([0, 0, 0, 0])

    model = MLP(layer_sizes=[3, 4, 1])
    model.train(X, y, n_iterations=10)

    preds = model.predict(X)

    assert preds.shape == (4,)


# test deterministic behavior (same init -> same result)
def test_deterministic():
    np.random.seed(42)
    X = np.random.rand(6, 2)
    y = np.array([0, 1, 0, 1, 0, 1])

    np.random.seed(42)
    model1 = MLP(layer_sizes=[2, 3, 1])
    model1.train(X, y, n_iterations=50)

    np.random.seed(42)
    model2 = MLP(layer_sizes=[2, 3, 1])
    model2.train(X, y, n_iterations=50)

    for w1, w2 in zip(model1.w_, model2.w_):
        assert np.allclose(w1, w2)