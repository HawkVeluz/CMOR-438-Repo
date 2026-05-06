import numpy as np
import pytest
from rice_ml.unsupervised_learning import PCA


# test output shape after transform
def test_transform_shape():
    X = np.random.rand(10, 5)

    model = PCA(n_components=2)
    X_transformed = model.train_transform(X)

    assert X_transformed.shape == (10, 2)


# test components shape
def test_components_shape():
    X = np.random.rand(8, 4)

    model = PCA(n_components=3)
    model.train(X)

    assert model.components_.shape == (4, 3)


# test explained variance ratio sums correctly
def test_explained_variance_ratio():
    X = np.random.rand(20, 6)

    model = PCA(n_components=3)
    model.train(X)

    total_ratio = np.sum(model.explained_variance_ratio_)
    assert 0 < total_ratio <= 1


# test orthogonality of components
def test_orthogonality():
    X = np.random.rand(15, 5)

    model = PCA(n_components=3)
    model.train(X)

    comps = model.components_

    # dot product between different components should be ~0
    dot_matrix = comps.T @ comps
    identity = np.eye(3)

    assert np.allclose(dot_matrix, identity, atol=1e-6)


# test variance decreases with components
def test_variance_order():
    X = np.random.rand(20, 4)

    model = PCA(n_components=3)
    model.train(X)

    ev = model.explained_variance_

    # should be sorted descending
    assert np.all(ev[:-1] >= ev[1:])


# test reconstruction (approximate)
def test_reconstruction():
    X = np.random.rand(10, 3)

    model = PCA(n_components=2)
    model.train(X)

    X_proj = model.transform(X)
    X_recon = X_proj @ model.components_.T + model.mean_

    # reconstruction error should be finite and reasonable
    error = np.mean((X - X_recon) ** 2)
    assert error >= 0
    assert np.isfinite(error)


# test identical points
def test_identical_points():
    X = np.array([[1, 2, 3]] * 5)

    model = PCA(n_components=2)
    model.train(X)

    # no variance
    assert np.allclose(model.explained_variance_, 0)


# test single component
def test_single_component():
    X = np.random.rand(10, 4)

    model = PCA(n_components=1)
    X_transformed = model.train_transform(X)

    assert X_transformed.shape == (10, 1)


# test deterministic behavior
def test_deterministic():
    X = np.random.rand(12, 3)

    model1 = PCA(n_components=2)
    model2 = PCA(n_components=2)

    model1.train(X)
    model2.train(X)

    assert np.allclose(model1.components_, model2.components_)
    assert np.allclose(model1.explained_variance_, model2.explained_variance_)