import numpy as np
from collections import deque

class DBSCAN:
    """Density-Based Spatial Clustering of Applications with Noise (DBSCAN) implementation.
    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for them to be considered as in the same neighborhood.
    min_samples : int, default=5
        The number of samples in a neighborhood for a point to be considered as a core point.
    """
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def train(self, X):
        """ Train DBSCAN model on given data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to train the model on.
        Attributes
        ----------
        labels_: ndarray of shape (n_samples,)
            The cluster labels for each point in the dataset. Noisy samples are given the label -1.
        """
        X = np.array(X, dtype=float)
        n_samples = X.shape[0]

        # -1 = noise, 0 = unvisited, >=1 = cluster id
        labels = np.zeros(n_samples, dtype=int)
        cluster_id = 0

        for i in range(n_samples):
            if labels[i] != 0:
                continue

            # find neighbors
            neighbors = self.region_query (X, i)

            if len(neighbors) < self.min_samples:
                labels[i] = -1
                continue

            # start new cluster
            cluster_id += 1
            labels[i] = cluster_id

            # queue for BFS expansion
            queue = deque(neighbors)

            while queue:
                j = queue.popleft()

                if labels[j] == -1:
                    labels[j] = cluster_id

                if labels[j] != 0:
                    continue

                labels[j] = cluster_id

                # find neighbors of j
                neighbors_j = self.region_query(X, j)

                if len(neighbors_j) >= self.min_samples:
                    queue.extend(neighbors_j)

        self.labels_ = labels
        return self

    def region_query(self, X, idx):
        """
        Find all points within the epsilon neighborhood of a given point.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        idx : int
            The index of the point for which to find neighbors.
        Returns
        -------
        ndarray of shape (n_neighbors,)
            The indices of points within the epsilon neighborhood.
        """
        distances = np.linalg.norm(X - X[idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def predict(self, X):
        """
        Train model and return cluster labels.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to train the model on.
        Attributes
        ----------
        labels_: ndarray of shape (n_samples,)
            The cluster labels for each point in the dataset.
        """
        self.train(X)
        return self.labels_