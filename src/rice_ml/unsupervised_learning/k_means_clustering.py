import numpy as np

class KMeansClustering:
    """
    K-means clustering algorithm for unsupervised learning.
    Parameters
    ----------
    k: int
        Number of clusters to form.
    max_iters: int
        Maximum number of iterations to run the algorithm.
    tol: float
        Tolerance for convergence (based on centroid movement).
    random_state: int or None
        Random seed for reproducibility.

    Attributes
    ----------
    centroids_: array, shape (k, n_features)
        Coordinates of cluster centers.
    labels_: array, shape (n_samples,)
        Index of the cluster each sample belongs to.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
    """
    def __init__(self, k=3, max_iters=100, tol=1e-4, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def train(self, X):
        """
        Train the K-means algorithm on the provided data.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = np.array(X, dtype=float)
        n_samples, n_features = X.shape

        rng = np.random.default_rng(self.random_state)

        # random centroids
        idx = rng.choice(n_samples, self.k, replace=False)
        self.centroids = X[idx]

        for _ in range(self.max_iters):
            # assign clusters
            distances = np.linalg.norm(X[:, None] - self.centroids[None, :], axis=2)
            labels = np.argmin(distances, axis=1)

            # new centroids
            new_centroids = np.zeros_like(self.centroids)

            for i in range(self.k):
                points = X[labels == i]

                if len(points) == 0:
                    new_centroids[i] = X[rng.integers(0, n_samples)]
                else:
                    new_centroids[i] = np.mean(points, axis=0)

            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids

            if shift < self.tol:
                break

        self.labels_ = labels
        return self

    def predict(self, X):
        """Predict the target clusters for the provided data.
        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        
        Attributes
        -------
        pred: ndarray of shape (n_samples,)
        Predicted cluster for each data point in X.
        """
        X = np.array(X, dtype=float)

        distances = np.linalg.norm(X[:, None] - self.centroids[None, :], axis=2)
        pred = np.argmin(distances, axis=1)
        return pred
    
    def inertia(self, X):
        """
        Calculate the inertia (sum of squared distances to closest centroid) for the provided data.
        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        Attributes
        -------
        inertia_: float
            Sum of squared distances of samples to their closest cluster center.
        """
        X = np.array(X, dtype=float)

        if self.centroids is None or self.labels_ is None:
            raise ValueError("Model must be fitted before computing inertia.")

        distances = np.linalg.norm(X - self.centroids[self.labels_], axis=1) ** 2
        self.inertia_ = np.sum(distances)
        return self.inertia_