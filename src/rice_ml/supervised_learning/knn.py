import numpy as np

def distance(p, q):
    """
    Calculates the Euclidean distance between two points p and q.

        d(p, q) = sqrt((p - q)^T (p - q))

    Parameters
    ----------
    p : numpy array — first point
    q : numpy array — second point

    Returns
    -------
    float — the distance between p and q
    """
    return np.sqrt((p - q) @ (p - q))

class KNN(object):
    """
    K-Nearest Neighbors classifier.

    Stores the training data and uses it to predict labels for new data points 
    based on the majority label among the k closest neighbors.

    Parameters
    ----------
    k: int, default=3
        Number of nearest neighbors to use.

    Attributes
    ----------
    X_train: ndarray of shape (n_samples, n_features)
        Training feature matrix.
    y_train: ndarray of shape (n_samples)
        Training labels.
    """
    def __init__(self, k=3):
        self.k = k

    def train(self, X, y):
        """
        Store the training data (no computation performed).

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y)
        return self
    def predict(self, X):
        """
        Predict the class labels for the provided data.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred: ndarray of shape (n_samples,)
            Predicted class labels for each data point in X.
        """
        X = np.array(X, dtype=float)
        y_pred = []
        for x in X:
            # Compute distances from x to all training points
            distances = np.linalg.norm(self.X_train - x, axis=1)
            # Find the indices of the k nearest neighbors
            neighbor_indices = np.argsort(distances)[:self.k]
            # Get the labels of the k nearest neighbors
            neighbor_labels = self.y_train[neighbor_indices]
            # Determine the majority label among the neighbors
            values, counts = np.unique(neighbor_labels, return_counts=True)
            majority_label = values[np.argmax(counts)]
            # Classify x as the majority label
            y_pred.append(majority_label)
        return np.array(y_pred)
    
class KNNRegressor(object):
    """
    K-Nearest Neighbors regressor.

    Stores the training data and uses it to predict continuous values for new 
    data points based on the average value among the k closest neighbors.

    Parameters
    ----------
    k: int, default=3
        Number of nearest neighbors to use.

    Attributes
    ----------
    X_train: ndarray of shape (n_samples, n_features)
        Training feature matrix.
    y_train: ndarray of shape (n_samples)
        Training target values.
    """
    def __init__(self, k=3):
        self.k = k

    def train(self, X, y):
        """
        Store the training data (no computation performed).

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y, dtype=float)
        return self
    
    def predict(self, X):
        """Predict the target values for the provided data.
        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        
        Attributes
        -------
        y_pred: ndarray of shape (n_samples,)
        Predicted target values for each data point in X.
        """
        X = np.array(X, dtype=float)
        y_pred = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            neighbor_indices = np.argsort(distances)[:self.k]
            neighbor_values = self.y_train[neighbor_indices]
            # Predict the value for x as the average of the neighbor values
            average_value = np.mean(neighbor_values)
            y_pred.append(average_value)
        return np.array(y_pred)