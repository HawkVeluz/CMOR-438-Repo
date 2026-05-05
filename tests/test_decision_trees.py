import numpy as np
from rice_ml.supervised_learning import DecisionTree, DecisionTreeRegressor

# Decision Tree Classifier initialization
def test_tree_classifier_initialization():
    model = DecisionTree(max_depth=5)
    assert model.max_depth == 5

# Decision Tree Classifier shape consistency
def test_tree_classifier_shape():
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 0, 1, 1])

    model = DecisionTree(max_depth=3)
    model.train(X, y)

    preds = model.predict(X)

    assert preds.shape == y.shape

# Decision Tree Classifier perfect separable data
def test_tree_classifier_perfect_split():
    X = np.array([[0], [1], [10], [11]])
    y = np.array([0, 0, 1, 1])

    model = DecisionTree(max_depth=3)
    model.train(X, y)

    preds = model.predict(X)

    assert np.array_equal(preds, y)

# Decision Tree Classifier binary output only
def test_tree_classifier_binary_output():
    X = np.random.rand(50, 3)
    y = (X[:, 0] > 0.5).astype(int)

    model = DecisionTree(max_depth=5)
    model.train(X, y)

    preds = model.predict(X)

    assert set(np.unique(preds)).issubset({0, 1})

# Decision Tree Classifier no NaN outputs
def test_tree_classifier_no_nan():
    X = np.random.rand(40, 2)
    y = np.random.randint(0, 2, size=40)

    model = DecisionTree(max_depth=5)
    model.train(X, y)

    preds = model.predict(X)

    assert np.all(np.isfinite(preds))

# Decision Tree Regressor initialization
def test_tree_regressor_initialization():
    model = DecisionTreeRegressor(max_depth=5)
    assert model.max_depth == 5

# Decision Tree Regressor shape consistency
def test_tree_regressor_shape():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([1.5, 2.5, 3.5, 4.5])

    model = DecisionTreeRegressor(max_depth=3)
    model.train(X, y)

    preds = model.predict(X)

    assert preds.shape == y.shape

# Decision Tree Regressor perfect fit on simple data
def test_tree_regressor_perfect_fit():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2.0, 4.0, 6.0, 8.0])

    model = DecisionTreeRegressor(max_depth=3)
    model.train(X, y)

    preds = model.predict(X)

    assert np.allclose(preds, y, atol=1.0)

# Decision Tree Regressor no NaN outputs
def test_tree_regressor_no_nan():
    X = np.random.rand(50, 3)
    y = np.random.rand(50)

    model = DecisionTreeRegressor(max_depth=5)
    model.train(X, y)

    preds = model.predict(X)

    assert np.all(np.isfinite(preds))
