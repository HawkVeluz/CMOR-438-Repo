import numpy as np


class DecisionTree:
    """A simple implementation of a Decision Tree Regressor for regression tasks.
    The tree is built by recursively splitting the data based on feature thresholds that minimize the mean squared error (MSE).
    Parameters:
    - max_depth: The maximum depth of the tree. Default is 5.
    - min_samples_split: The minimum number of samples required to split an internal node. Default is 2.
    Methods:
    - train(X, y): Trains the decision tree regressor on the provided data.
    - predict(X): Predicts the target values for the given input data.
    - grow_tree(X, y, depth): Recursively builds the decision tree.
    - best_split(X, y, n_features): Finds the best feature and threshold for splitting the data.
    """
    class Node:
        """
        A node in the decision tree.
        Attributes:
        - feature: The index of the feature used for splitting.
        - threshold: The threshold value for the split.
        - left: The left child node.
        - right: The right child node.
        - value: The predicted value if the node is a leaf.
        - gini: The gini impurity of the node.
        - n_samples: The number of samples at the node.
        """
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, gini=None, n_samples=None):
            self.feature = feature      # index of feature to split on
            self.threshold = threshold  # split value
            self.left = left            # left child
            self.right = right          # right child
            self.value = value          # class label if leaf
            self.gini = gini            # gini impurity of the node
            self.n_samples = n_samples  # number of samples at the node

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def train(self, X, y):
        """
        Train the decision tree regressor on the provided data.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.array(X)
        y = np.array(y)
        self.root = self.grow_tree(X, y, depth=0)
    
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
        X = np.array(X)
        return np.array([self.traverse(x, self.root) for x in X])
    
    def grow_tree(self, X, y, depth):
        """
        Recursively build the decision tree.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)
        depth: int
            Current depth of the tree.

        Attributes
        -------
        Node
            The root node of the subtree built from the provided data.       
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping conditions
        if (depth >= self.max_depth or
            n_labels == 1 or
            n_samples < self.min_samples_split):
            leaf_value = self.most_common_label(y)
            return self.Node(value=leaf_value)

        # find best split
        best_feature, best_threshold = self.best_split(X, y, n_features)

        if best_feature is None:
            return self.Node(value=self.most_common_label(y))

        # split
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = X[:, best_feature] >= best_threshold

        left = self.grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self.grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return self.Node(best_feature, best_threshold, left, right)
    
    def best_split(self, X, y, n_features):
        """
        Find the best feature and threshold for splitting the data.
        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)
        n_features: int
            The number of features in the dataset.

        Returns
        -------
        best_feature: int
            The index of the best feature for splitting.
        best_threshold: float
            The threshold value for the best split.
        """
        best_gain = -np.inf
        split_idx, split_thresh = None, None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for thresh in thresholds:
                gain = self.information_gain(y, X[:, feature], thresh)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_thresh = thresh

        return split_idx, split_thresh
    
    def ginicalc(self, y):
        """
        Calculates Gini impurity.
        Parameters
        ----------
        y: array-like of shape (n_samples,)

        Returns
        -------
        gini: float
            The Gini impurity of the labels.
        """
        if len(y) == 0:
            return 0 # avoid division by zero
        values, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def information_gain(self, y, X_column, split_thresh):
        """
        Calculates the information gain of a potential split.
        Parameters
        ----------
        y: array-like of shape (n_samples,)
        X_column: array-like of shape (n_samples,)
        split_thresh: float

        Returns
        -------
        gain: float
            The information gain of the potential split.
        """
        parent_gini = self.ginicalc(y)

        left_idxs = X_column < split_thresh
        right_idxs = X_column >= split_thresh


        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(y[left_idxs]), len(y[right_idxs])

        gini_l = self.ginicalc(y[left_idxs])
        gini_r = self.ginicalc(y[right_idxs])

        child_gini = (n_l / n) * gini_l + (n_r / n) * gini_r

        return parent_gini - child_gini
    
    def most_common_label(self, y):
        """
        Finds the most common label in the provided array.
        Parameters
        ----------
        y: array-like of shape (n_samples,)
        Returns
        -------
        most_common: int or str
        """
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]
    
    def traverse(self, x, node):
        """
        Traverses the tree to make a prediction for a single input sample.
        Parameters
        ----------
        x: array-like of shape (n_features,)
        node: Node
            The current node in the tree.

        Returns
        -------
        prediction: float
            The predicted value for the input sample.
        """
        if node.value is not None:
            return node.value

        if x[node.feature] < node.threshold:
            return self.traverse(x, node.left)
        return self.traverse(x, node.right)
    

import numpy as np


class DecisionTreeRegressor:
    """A simple implementation of a Decision Tree Regressor for regression tasks.
    The tree is built by recursively splitting the data based on feature thresholds that minimize the mean squared error (MSE).
    Parameters:
    - max_depth: The maximum depth of the tree. Default is 5.
    - min_samples_split: The minimum number of samples required to split an internal node. Default is 2.
    Methods:
    - train(X, y): Trains the decision tree regressor on the provided data.
    - predict(X): Predicts the target values for the given input data.
    - grow_tree(X, y, depth): Recursively builds the decision tree.
    - best_split(X, y, n_features): Finds the best feature and threshold for splitting the data.
    - mse_calc(y): Calculates the mean squared error of the target values.
    - traverse(x, node): Traverses the tree to make a prediction for a single input sample.
    """

    class Node:
        """
        A node in the decision tree.
        Attributes:
        - feature: The index of the feature used for splitting.
        - threshold: The threshold value for the split.
        - left: The left child node.
        - right: The right child node.
        - value: The predicted value if the node is a leaf.
        - mse: The mean squared error of the node (used for regression).
        - n_samples: The number of samples at the node.
        """
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, mse=None, n_samples=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value          
            self.mse = mse              
            self.n_samples = n_samples

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def train(self, X, y):
        """
        Train the decision tree regressor on the provided data.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        self.root = self.grow_tree(X, y, depth=0)


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
        return np.array([self.traverse(x, self.root) for x in X])


    def grow_tree(self, X, y, depth):
        """
        Recursively build the decision tree.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)
        depth: int
            Current depth of the tree.

        Attributes
        -------
        Node
            The root node of the subtree built from the provided data.       
        """
        n_samples, n_features = X.shape
        mse = self.mse_calc(y)

        # stopping conditions
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            np.var(y) == 0):
            return self.Node(value=np.mean(y), mse=mse, n_samples=n_samples)

        # find best split
        best_feature, best_threshold = self.best_split(X, y, n_features)

        if best_feature is None:
            return self.Node(value=np.mean(y), mse=mse, n_samples=n_samples)

        # split
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = X[:, best_feature] >= best_threshold

        left = self.grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self.grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return self.Node(best_feature, best_threshold, left, right, mse=mse, n_samples=n_samples)

    def best_split(self, X, y, n_features):
        """
        Find the best feature and threshold for splitting the data.
        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)
        n_features: int
            The number of features in the dataset.

        Returns
        -------
        best_feature: int
            The index of the best feature for splitting.
        best_threshold: float
            The threshold value for the best split.
        """
        best_gain = -np.inf
        split_idx, split_thresh = None, None

        parent_mse = self.mse_calc(y)

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for thresh in thresholds:
                left_idxs = X[:, feature] < thresh
                right_idxs = X[:, feature] >= thresh

                if left_idxs.sum() == 0 or right_idxs.sum() == 0:
                    continue

                n = len(y)
                n_l, n_r = left_idxs.sum(), right_idxs.sum()

                mse_l = self.mse_calc(y[left_idxs])
                mse_r = self.mse_calc(y[right_idxs])

                child_mse = (n_l / n) * mse_l + (n_r / n) * mse_r
                gain = parent_mse - child_mse

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_thresh = thresh

        return split_idx, split_thresh

    def mse_calc(self, y):
        """
        Calculates mean squared error.
        Parameters
        ----------
        y: array-like of shape (n_samples,)

        Returns
        -------
        mse: float
            The mean squared error of the labels.
        """
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)

    def traverse(self, x, node):
        """
        Traverses the tree to make a prediction for a single input sample.
        Parameters
        ----------
        x: array-like of shape (n_features,)
        node: Node
            The current node in the tree.

        Returns
        -------
        prediction: float
            The predicted value for the input sample.
        """
        if node.value is not None:
            return node.value

        if x[node.feature] < node.threshold:
            return self.traverse(x, node.left)
        return self.traverse(x, node.right)