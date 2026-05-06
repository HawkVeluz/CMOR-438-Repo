import numpy as np
import pytest
from rice_ml.supervised_learning import (
    BaggingClassifier,
    BaggingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)


# bagging classifier

def test_bagging_classifier_basic():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])

    model = BaggingClassifier(n_estimators=5, max_depth=3)
    model.train(X, y)
    preds = model.predict(X)

    assert preds.shape == (4,)
    assert set(preds).issubset({0, 1})


def test_bagging_classifier_consistency():
    np.random.seed(42)

    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, size=10)

    model1 = BaggingClassifier(n_estimators=5)
    model2 = BaggingClassifier(n_estimators=5)

    np.random.seed(42)
    model1.train(X, y)

    np.random.seed(42)
    model2.train(X, y)

    preds1 = model1.predict(X)
    preds2 = model2.predict(X)

    assert np.array_equal(preds1, preds2)


# baggging regressor

def test_bagging_regressor_basic():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0.0, 1.0, 2.0, 3.0])

    model = BaggingRegressor(n_estimators=5, max_depth=3)
    model.train(X, y)
    preds = model.predict(X)

    assert preds.shape == (4,)
    assert np.all(np.isfinite(preds))


def test_bagging_regressor_average_behavior():
    X = np.array([[0], [1], [2]])
    y = np.array([1.0, 1.0, 1.0])

    model = BaggingRegressor(n_estimators=5)
    model.train(X, y)
    preds = model.predict(X)

    # should predict close to constant
    assert np.allclose(preds, 1.0, atol=1e-1)


#rfc

def test_random_forest_classifier_basic():
    X = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1]
    ])
    y = np.array([0, 0, 1, 1])

    model = RandomForestClassifier(n_estimators=5, max_depth=3)
    model.train(X, y)
    preds = model.predict(X)

    assert preds.shape == (4,)
    assert set(preds).issubset({0, 1})


def test_random_forest_classifier_feature_subsampling():
    X = np.random.rand(20, 5)
    y = np.random.randint(0, 2, size=20)

    model = RandomForestClassifier(n_estimators=5, max_features=2)
    model.train(X, y)

    for _, feat_idxs in model.models:
        assert len(feat_idxs) == 2


def test_random_forest_classifier_consistency():
    np.random.seed(42)

    X = np.random.rand(15, 3)
    y = np.random.randint(0, 2, size=15)

    model1 = RandomForestClassifier(n_estimators=5)
    model2 = RandomForestClassifier(n_estimators=5)

    np.random.seed(42)
    model1.train(X, y)

    np.random.seed(42)
    model2.train(X, y)

    preds1 = model1.predict(X)
    preds2 = model2.predict(X)

    assert np.array_equal(preds1, preds2)


# rfr

def test_random_forest_regressor_basic():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0.0, 1.0, 2.0, 3.0])

    model = RandomForestRegressor(n_estimators=5, max_depth=3)
    model.train(X, y)
    preds = model.predict(X)

    assert preds.shape == (4,)
    assert np.all(np.isfinite(preds))


def test_random_forest_regressor_feature_subsampling():
    X = np.random.rand(20, 4)
    y = np.random.rand(20)

    model = RandomForestRegressor(n_estimators=5, max_features=2)
    model.train(X, y)

    for _, feat_idxs in model.models:
        assert len(feat_idxs) == 2


def test_random_forest_regressor_average_behavior():
    X = np.array([[0], [1], [2]])
    y = np.array([2.0, 2.0, 2.0])

    model = RandomForestRegressor(n_estimators=5)
    model.train(X, y)
    preds = model.predict(X)

    assert np.allclose(preds, 2.0, atol=1e-1)