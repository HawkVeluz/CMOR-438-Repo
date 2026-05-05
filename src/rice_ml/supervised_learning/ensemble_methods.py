import numpy as np
from .decision_trees import DecisionTree, DecisionTreeRegressor

class BaggingRegressor:
    def __init__(self, base_model, n_estimators=10, max_samples=1.0, random_state=None):
        """
        base_model: uninitialized model class (e.g., DecisionTreeRegressor)
        n_estimators: number of models in ensemble
        max_samples: fraction of dataset used per model
        """
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.models = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_samples = X.shape[0]
        sample_size = int(self.max_samples * n_samples)

        rng = np.random.default_rng(self.random_state)

        self.models = []

        for _ in range(self.n_estimators):
            indices = rng.choice(n_samples, size=sample_size, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            model = self.base_model()
            model.train(X_sample, y_sample)
            self.models.append(model)

        return self

    def predict(self, X):
        X = np.array(X)

        preds = np.array([model.predict(X) for model in self.models])
        return np.mean(preds, axis=0)


class RandomForestRegressor:
    """Random Forest Regressor using Decision Trees as base models.
    Parameters
    ----------
    n_estimators: int
    max_depth: int or None, 
    min_samples_split: int
    max_features: int or None
    random_state: int or None
    Attributes
    ----------
    trees_: list of DecisionTreeRegressor
    """
    def __init__(self, DecisionTreeRegressor,n_estimators=10, max_samples=1.0, max_features=None, random_state=None):
        self.DecisionTreeRegressor = DecisionTreeRegressor
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        self.models = []
        self.feature_indices = []

    def train(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        sample_size = int(self.max_samples * n_samples)

        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        rng = np.random.default_rng(self.random_state)

        self.models = []
        self.feature_indices = []

        for _ in range(self.n_estimators):
            # bootstrap samples
            sample_idx = rng.choice(n_samples, size=sample_size, replace=True)

            # random feature subset
            feat_idx = rng.choice(n_features, size=self.max_features, replace=False)

            X_sample = X[sample_idx][:, feat_idx]
            y_sample = y[sample_idx]

            model = self.base_model()
            model.train(X_sample, y_sample)

            self.models.append(model)
            self.feature_indices.append(feat_idx)

        return self

    def predict(self, X):
        X = np.array(X)

        preds = []

        for model, feat_idx in zip(self.models, self.feature_indices):
            X_subset = X[:, feat_idx]
            preds.append(model.predict(X_subset))

        return np.mean(preds, axis=0)