import numpy as np
from rice_ml.supervised_learning import DecisionTree
from rice_ml.supervised_learning import DecisionTreeRegressor

# bagging

class BaggingClassifier:
    """
    Simple bagging ensemble method for classification using decision trees.
    Parameters:
    - n_estimators: number of trees in the ensemble
    - max_depth: maximum depth of each decision tree
    Attributes:
    - models: list of trained decision tree models
    """
    def __init__(self, n_estimators=10, max_depth=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models = []

    def train(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples = X.shape[0]

        self.models = []

        for _ in range(self.n_estimators):
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[idxs]
            y_sample = y[idxs]

            model = DecisionTree(max_depth=self.max_depth)
            model.train(X_sample, y_sample)
            self.models.append(model)

        return self

    def predict(self, X):
        X = np.array(X)

        pred = np.array([model.predict(X) for model in self.models])
        pred = pred.T 

        final_pred = []
        for row in pred:
            values, counts = np.unique(row, return_counts=True)
            final_pred.append(values[np.argmax(counts)])

        return np.array(final_pred)


class BaggingRegressor:
    """
    Simple bagging ensemble method for regression using decision trees.
    Parameters:
    - n_estimators: number of trees in the ensemble
    - max_depth: maximum depth of each decision tree
    Attributes:
    - models: list of trained decision tree models
    """
    def __init__(self, n_estimators=10, max_depth=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models = []

    def train(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples = X.shape[0]

        self.models = []

        for _ in range(self.n_estimators):
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[idxs]
            y_sample = y[idxs]

            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.train(X_sample, y_sample)
            self.models.append(model)

        return self

    def predict(self, X):
        X = np.array(X)

        pred = np.array([model.predict(X) for model in self.models])
        return np.mean(pred, axis=0)


# random forests

class RandomForestClassifier:
    """
    Random Forest classifier using decision trees as base learners.
    Parameters:
    - n_estimators: number of trees in the ensemble
    - max_depth: maximum depth of each decision tree
    - max_features: number of features to consider when looking for the best split
    Attributes:
    - models: list of trained decision tree models and their corresponding feature indices
    """
    def __init__(self, n_estimators=10, max_depth=5, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.models = []

    def train(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape

        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        self.models = []

        for _ in range(self.n_estimators):
            # bootstrap sampling
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[idxs]
            y_sample = y[idxs]

            # feature subsampling
            feat_idxs = np.random.choice(n_features, self.max_features, replace=False)

            model = DecisionTree(max_depth=self.max_depth)
            model.train(X_sample[:, feat_idxs], y_sample)

            self.models.append((model, feat_idxs))

        return self

    def predict(self, X):
        X = np.array(X)

        preds = []
        for model, feat_idxs in self.models:
            pred = model.predict(X[:, feat_idxs])
            preds.append(pred)

        preds = np.array(preds).T  
        final_preds = []
        for row in preds:
            values, counts = np.unique(row, return_counts=True)
            final_preds.append(values[np.argmax(counts)])

        return np.array(final_preds)


class RandomForestRegressor:
    """
    Random Forest regressor using decision trees as base learners.
    Parameters:
    - n_estimators: number of trees in the ensemble
    - max_depth: maximum depth of each decision tree
    - max_features: number of features to consider when looking for the best split
    Attributes:
    - models: list of trained decision tree models and their corresponding feature indices
    """
    def __init__(self, n_estimators=10, max_depth=5, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.models = []

    def train(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape

        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        self.models = []

        for _ in range(self.n_estimators):
            # bootstrap sampling
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[idxs]
            y_sample = y[idxs]

            # feature subsampling
            feat_idxs = np.random.choice(0, self.max_features, replace=False)

            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.train(X_sample[:, feat_idxs], y_sample)

            self.models.append((model, feat_idxs))

        return self

    def predict(self, X):
        X = np.array(X)

        preds = []
        for model, feat_idxs in self.models:
            pred = model.predict(X[:, feat_idxs])
            preds.append(pred)

        preds = np.array(preds)
        return np.mean(preds, axis=0)