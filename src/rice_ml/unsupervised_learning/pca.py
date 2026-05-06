import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA) for dimensionality reduction.
    Parameters
    ----------
    n_components: int
        The number of principal components to compute.
    Attributes
    ----------
    components_: array, shape (n_components, n_features)
        The principal components.
    mean_: array, shape (n_features,)
        The mean of the training data.
    explained_variance_: array, shape (n_components,)
        The variance explained by each principal component.
    totalvariance_: float
        The total variance in the data.
    explained_variance_ratio_: array, shape (n_components,)
        The percentage of variance explained by each of the selected components.
    """
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.totalvariance_ = None
        self.explained_variance_ratio_ = None

    def train(self, X):
        X = np.array(X, dtype=float)

        # center data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # compute covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # sort eigenvalues and eigenvectors (descending)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # select top components
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.totalvariance_ = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / self.totalvariance_
        return self

    def transform(self, X):
        """
        Project the data onto the principal components.
        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input data to transform.
        Attributes
        ----------
        X_transformed: array of shape (n_samples, n_components)
            The input data projected onto the top principal components."""
        X = np.array(X, dtype=float)
        X_centered = X - self.mean_

        # Project onto principal components
        return X_centered @ self.components_

    def train_transform(self, X):
        self.train(X)
        return self.transform(X)