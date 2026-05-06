import numpy as np
import pytest
from rice_ml.unsupervised_learning import CommunityDetection


# test basic two-community separation
def test_two_communities():
    # adjacency matrix with two clear clusters
    A = np.array([
        [0,1,1,0,0,0],
        [1,0,1,0,0,0],
        [1,1,0,0,0,0],
        [0,0,0,0,1,1],
        [0,0,0,1,0,1],
        [0,0,0,1,1,0],
    ])

    model = CommunityDetection()
    labels = model.predict(A)

    # should detect 2 communities
    assert model.n_communities_ == 2

    # first 3 nodes same community
    assert len(set(labels[:3])) == 1

    # last 3 nodes same community
    assert len(set(labels[3:])) == 1


# test single community (fully connected graph)
def test_single_community():
    A = np.ones((5,5)) - np.eye(5)

    model = CommunityDetection()
    labels = model.predict(A)

    assert model.n_communities_ == 1
    assert len(set(labels)) == 1


# test isolated nodes
def test_isolated_nodes():
    A = np.zeros((4,4))

    model = CommunityDetection()
    labels = model.predict(A)

    # each node remains its own community
    assert model.n_communities_ == 4
    assert len(set(labels)) == 4


# test weighted graph influence
def test_weighted_graph():
    A = np.array([
        [0,5,1,0],
        [5,0,1,0],
        [1,1,0,5],
        [0,0,5,0],
    ])

    model = CommunityDetection()
    labels = model.predict(A)

    # expect two clusters: (0,1) and (2,3)
    assert model.n_communities_ == 2

    assert labels[0] == labels[1]
    assert labels[2] == labels[3]


# test convergence before max_iters
def test_convergence():
    A = np.array([
        [0,1,1],
        [1,0,1],
        [1,1,0],
    ])

    model = CommunityDetection(max_iters=50)
    labels = model.predict(A)

    # should converge to one community
    assert model.n_communities_ == 1


# test label shape
def test_output_shape():
    A = np.array([
        [0,1],
        [1,0]
    ])

    model = CommunityDetection()
    labels = model.predict(A)

    assert labels.shape == (2,)