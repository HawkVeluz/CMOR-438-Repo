import numpy as np
import pytest
from rice_ml.unsupervised_learning import DBSCAN


# test simple two clusters
def test_two_clusters():
    X = np.array([
        [0, 0], [0.1, 0.1], [0.2, 0.2],
        [5, 5], [5.1, 5.1], [5.2, 5.2]
    ])

    model = DBSCAN(eps=0.5, min_samples=2)
    labels = model.predict(X)

    unique_clusters = set(labels)
    unique_clusters.discard(-1)

    assert len(unique_clusters) == 2


# test noise points
def test_noise_points():
    X = np.array([
        [0, 0], [0.1, 0.1],
        [10, 10]  # isolated point
    ])

    model = DBSCAN(eps=0.3, min_samples=2)
    labels = model.predict(X)

    # last point should be noise
    assert labels[-1] == -1


# test all noise
def test_all_noise():
    X = np.array([
        [0, 0],
        [5, 5],
        [10, 10]
    ])

    model = DBSCAN(eps=0.1, min_samples=2)
    labels = model.predict(X)

    assert np.all(labels == -1)


# test single cluster
def test_single_cluster():
    X = np.array([
        [1, 1], [1.1, 1.1], [0.9, 0.9]
    ])

    model = DBSCAN(eps=0.5, min_samples=2)
    labels = model.predict(X)

    unique_clusters = set(labels)
    unique_clusters.discard(-1)

    assert len(unique_clusters) == 1


# test border point assignment
def test_border_point():
    X = np.array([
        [0, 0], [0.1, 0.1], [0.2, 0.2],  # core cluster
        [0.4, 0.4]  # border point
    ])

    model = DBSCAN(eps=0.5, min_samples=3)
    labels = model.predict(X)

    # border point should belong to cluster
    assert labels[-1] != -1


# test label shape
def test_output_shape():
    X = np.array([
        [0, 0],
        [1, 1]
    ])

    model = DBSCAN(eps=0.5, min_samples=1)
    labels = model.predict(X)

    assert labels.shape == (2,)


# test deterministic behavior (same input → same output)
def test_deterministic():
    X = np.array([
        [0, 0], [0.1, 0.1],
        [5, 5], [5.1, 5.1]
    ])

    model = DBSCAN(eps=0.5, min_samples=2)
    labels1 = model.predict(X)
    labels2 = model.predict(X)

    assert np.array_equal(labels1, labels2)