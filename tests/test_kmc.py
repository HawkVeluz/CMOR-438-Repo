import numpy as np
import pytest
from rice_ml.unsupervised_learning import KMeansClustering


# test two well-separated clusters
def test_two_clusters():
    X = np.array([
        [0, 0], [0.1, 0.1], [0.2, 0.2],
        [5, 5], [5.1, 5.1], [5.2, 5.2]
    ])

    model = KMeansClustering(k=2, random_state=42)
    model.train(X)

    labels = model.labels_
    assert len(np.unique(labels)) == 2


# test centroid shape
def test_centroid_shape():
    X = np.random.rand(10, 3)

    model = KMeansClustering(k=3, random_state=0)
    model.train(X)

    assert model.centroids.shape == (3, 3)


# test prediction shape
def test_predict_shape():
    X = np.random.rand(8, 2)

    model = KMeansClustering(k=2, random_state=0)
    model.train(X)

    preds = model.predict(X)
    assert preds.shape == (8,)


# test inertia is non-negative
def test_inertia_positive():
    X = np.random.rand(10, 2)

    model = KMeansClustering(k=2, random_state=0)
    model.train(X)

    inertia = model.inertia(X)
    assert inertia >= 0


# test convergence (centroids stabilize)
def test_convergence():
    X = np.array([
        [1, 1], [1.1, 1.1], [0.9, 0.9],
        [5, 5], [5.1, 5.1], [4.9, 4.9]
    ])

    model = KMeansClustering(k=2, max_iters=100, tol=1e-4, random_state=0)
    model.train(X)

    # run again and check centroids don't change much
    old_centroids = model.centroids.copy()
    model.train(X)
    new_centroids = model.centroids

    assert np.allclose(old_centroids, new_centroids, atol=1e-2)


# test identical points
def test_identical_points():
    X = np.array([[2, 2]] * 5)

    model = KMeansClustering(k=1, random_state=0)
    model.train(X)

    assert np.allclose(model.centroids[0], [2, 2])


# test k equals number of samples
def test_k_equals_n():
    X = np.random.rand(5, 2)

    model = KMeansClustering(k=5, random_state=0)
    model.train(X)

    assert len(np.unique(model.labels_)) == 5


# test deterministic with random_state
def test_deterministic():
    X = np.random.rand(20, 2)

    model1 = KMeansClustering(k=3, random_state=42)
    model2 = KMeansClustering(k=3, random_state=42)

    model1.train(X)
    model2.train(X)

    assert np.allclose(model1.centroids, model2.centroids)


# test empty cluster handling
def test_empty_cluster_reassignment():
    # crafted dataset where empty cluster is likely
    X = np.array([
        [0, 0], [0.1, 0.1],
        [10, 10], [10.1, 10.1]
    ])

    model = KMeansClustering(k=3, random_state=0)
    model.train(X)

    # should still have valid centroids
    assert model.centroids.shape == (3, 2)