import numpy as np
from sklearn.tree import DecisionTreeClassifier

from baselines import (
    IMPUTE_BASELINES,
    baseline_rf_impute,
    baseline_rf_tree_impute,
    make_rf_imputer,
)


def test_rf_tree_impute_is_registered():
    assert "rf_tree_impute" in IMPUTE_BASELINES
    assert IMPUTE_BASELINES["rf_tree_impute"] is baseline_rf_tree_impute
    assert "rf_impute" in IMPUTE_BASELINES
    assert IMPUTE_BASELINES["rf_impute"] is baseline_rf_impute


def test_make_rf_imputer_uses_requested_random_forest_size():
    imp = make_rf_imputer(n_estimators=7, max_iter=2, random_state=11)

    assert imp.max_iter == 2
    assert imp.random_state == 11
    assert imp.estimator.n_estimators == 7
    assert imp.estimator.random_state == 11


def test_rf_tree_impute_predicts_with_missing_test_features():
    X_train = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.2, 0.1, 0.0],
        [0.8, 0.9, 1.0],
    ])
    y_train = np.array([0, 0, 1, 1, 0, 1])
    X_test_missing = np.array([
        [0.0, np.nan, 0.0],
        [1.0, np.nan, 1.0],
    ])

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)

    preds = baseline_rf_tree_impute(clf, X_train, X_test_missing)

    assert preds.shape == (2,)
    assert not np.isnan(preds.astype(float)).any()


def test_rf_impute_predicts_with_missing_test_features():
    X_train = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.2, 0.1, 0.0],
        [0.8, 0.9, 1.0],
    ])
    y_train = np.array([0, 0, 1, 1, 0, 1])
    X_test_missing = np.array([
        [0.0, np.nan, 0.0],
        [1.0, np.nan, 1.0],
    ])

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)

    preds = baseline_rf_impute(clf, X_train, X_test_missing)

    assert preds.shape == (2,)
    assert not np.isnan(preds.astype(float)).any()
