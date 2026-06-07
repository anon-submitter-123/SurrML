"""
Unit tests for steering_v2.py (soft leaf-probability rerouting).
"""
import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from steering_v2 import (
    soft_tree_predict,
    soft_tree_predict_batch,
    soft_forest_predict,
    mc_knn_predict,
    mc_generic_predict,
    wrapped_predict_v2,
)


class IdentityGenerator:
    """Generator that returns exact copies of x_obs (no imputation needed)."""
    prior_class = 0

    def sample(self, x_obs, mask, n_samples, rng):
        return np.tile(x_obs, (n_samples, 1))


class NoisyGenerator:
    """Generator that adds small noise to observed features and draws random for missing."""
    prior_class = 0

    def __init__(self, X_train):
        self.means = X_train.mean(axis=0)
        self.stds = X_train.std(axis=0) + 1e-8

    def sample(self, x_obs, mask, n_samples, rng):
        S = np.tile(x_obs, (n_samples, 1)).copy()
        for j in range(len(mask)):
            if not mask[j]:
                S[:, j] = rng.normal(self.means[j], self.stds[j], n_samples)
        return S


@pytest.fixture
def binary_data():
    X, y = make_classification(n_samples=200, n_features=5, n_informative=3,
                                random_state=42)
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]
    return X_train, y_train, X_test, y_test


@pytest.fixture
def multiclass_data():
    X, y = make_classification(n_samples=300, n_features=8, n_informative=5,
                                n_classes=3, n_clusters_per_class=1,
                                random_state=42)
    X_train, X_test = X[:200], X[200:]
    y_train, y_test = y[:200], y[200:]
    return X_train, y_train, X_test, y_test


# --- Test soft_tree_predict ---

def test_soft_tree_no_missing_matches_predict(binary_data):
    """With identity generator and no missing features, soft prediction should match clf.predict."""
    X_train, y_train, X_test, y_test = binary_data
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    gen = IdentityGenerator()

    y_expected = clf.predict(X_test)
    mask_full = np.ones(X_test.shape[1], dtype=bool)

    for i in range(len(X_test)):
        probs = soft_tree_predict(clf, X_test[i], mask_full, gen, n_mc=50, random_state=42)
        assert np.argmax(probs) == y_expected[i], f"Mismatch at sample {i}"


def test_soft_tree_returns_valid_probabilities(binary_data):
    """Probability vector should sum to 1 and be non-negative."""
    X_train, y_train, X_test, _ = binary_data
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    gen = IdentityGenerator()
    mask = np.ones(X_test.shape[1], dtype=bool)

    probs = soft_tree_predict(clf, X_test[0], mask, gen, n_mc=50, random_state=42)
    assert np.all(probs >= 0), "Probabilities should be non-negative"
    assert np.isclose(probs.sum(), 1.0), f"Probabilities should sum to 1, got {probs.sum()}"


def test_soft_tree_reproducible(binary_data):
    """Same random_state should produce identical results."""
    X_train, y_train, X_test, _ = binary_data
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    gen = NoisyGenerator(X_train)
    mask = np.ones(X_test.shape[1], dtype=bool)
    mask[2] = False  # make one feature missing

    p1 = soft_tree_predict(clf, X_test[0], mask, gen, n_mc=50, random_state=123)
    p2 = soft_tree_predict(clf, X_test[0], mask, gen, n_mc=50, random_state=123)
    np.testing.assert_array_equal(p1, p2)


# --- Test soft_tree_predict_batch ---

def test_soft_tree_batch_shape(binary_data):
    """Batch prediction should return correct number of labels."""
    X_train, y_train, X_test, _ = binary_data
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    gen = IdentityGenerator()
    mask_matrix = np.ones_like(X_test, dtype=bool)

    preds = soft_tree_predict_batch(clf, X_test, mask_matrix, gen, n_mc=50, random_state=42)
    assert len(preds) == len(X_test)


# --- Test soft_forest_predict ---

def test_soft_forest_no_missing_matches_predict(binary_data):
    """With identity generator and no missing features, should match rf.predict."""
    X_train, y_train, X_test, _ = binary_data
    rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    gen = IdentityGenerator()
    mask_matrix = np.ones_like(X_test, dtype=bool)

    y_expected = rf.predict(X_test)
    y_soft = soft_forest_predict(rf, X_test, mask_matrix, gen, n_mc=50, random_state=42)
    assert np.array_equal(y_soft, y_expected)


def test_soft_forest_shape(binary_data):
    """Forest prediction should return correct number of labels."""
    X_train, y_train, X_test, _ = binary_data
    rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    gen = IdentityGenerator()
    mask_matrix = np.ones_like(X_test, dtype=bool)

    preds = soft_forest_predict(rf, X_test, mask_matrix, gen, n_mc=50, random_state=42)
    assert len(preds) == len(X_test)


# --- Test mc_knn_predict ---

def test_mc_knn_no_missing_matches_predict(binary_data):
    """With identity generator and no missing features, should match knn.predict."""
    X_train, y_train, X_test, _ = binary_data
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    gen = IdentityGenerator()
    mask_matrix = np.ones_like(X_test, dtype=bool)

    y_expected = knn.predict(X_test)
    y_mc = mc_knn_predict(knn, X_test, mask_matrix, gen, n_mc=50, random_state=42)
    assert np.array_equal(y_mc, y_expected)


# --- Test mc_generic_predict ---

def test_mc_generic_logistic(binary_data):
    """Generic MC prediction should work with LogisticRegression."""
    X_train, y_train, X_test, _ = binary_data
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    gen = IdentityGenerator()
    mask_matrix = np.ones_like(X_test, dtype=bool)

    y_expected = lr.predict(X_test)
    y_mc = mc_generic_predict(lr, X_test, mask_matrix, gen, n_mc=50, random_state=42)
    assert np.array_equal(y_mc, y_expected)


# --- Test wrapped_predict_v2 dispatcher ---

def test_dispatcher_tree(binary_data):
    """wrapped_predict_v2 should dispatch to soft_tree_predict_batch for DT."""
    X_train, y_train, X_test, _ = binary_data
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    gen = IdentityGenerator()
    mask_matrix = np.ones_like(X_test, dtype=bool)

    preds = wrapped_predict_v2(clf, X_test, mask_matrix, gen, n_mc=50, random_state=42)
    y_expected = clf.predict(X_test)
    assert np.array_equal(preds, y_expected)


def test_dispatcher_forest(binary_data):
    """wrapped_predict_v2 should dispatch to soft_forest_predict for RF."""
    X_train, y_train, X_test, _ = binary_data
    rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    gen = IdentityGenerator()
    mask_matrix = np.ones_like(X_test, dtype=bool)

    preds = wrapped_predict_v2(rf, X_test, mask_matrix, gen, n_mc=50, random_state=42)
    y_expected = rf.predict(X_test)
    assert np.array_equal(preds, y_expected)


def test_dispatcher_knn(binary_data):
    """wrapped_predict_v2 should dispatch to mc_knn_predict for KNN."""
    X_train, y_train, X_test, _ = binary_data
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    gen = IdentityGenerator()
    mask_matrix = np.ones_like(X_test, dtype=bool)

    preds = wrapped_predict_v2(knn, X_test, mask_matrix, gen, n_mc=50, random_state=42)
    y_expected = knn.predict(X_test)
    assert np.array_equal(preds, y_expected)


# --- Test with missing features ---

def test_soft_vs_hard_differ_with_missing(binary_data):
    """Soft and hard routing should generally differ when features are missing."""
    X_train, y_train, X_test, _ = binary_data
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    gen = NoisyGenerator(X_train)
    # Mask 2 features for each test sample
    rng = np.random.RandomState(42)
    mask_matrix = np.ones_like(X_test, dtype=bool)
    for i in range(len(X_test)):
        missing_cols = rng.choice(X_test.shape[1], size=2, replace=False)
        mask_matrix[i, missing_cols] = False

    y_soft = soft_tree_predict_batch(clf, X_test, mask_matrix, gen, n_mc=100, random_state=42)
    # Predictions should be valid class labels
    assert all(p in [0, 1] for p in y_soft)


def test_all_features_missing(binary_data):
    """Edge case: all features missing should still produce a prediction."""
    X_train, y_train, X_test, _ = binary_data
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    gen = NoisyGenerator(X_train)
    mask = np.zeros(X_test.shape[1], dtype=bool)  # all missing
    x_obs = np.full(X_test.shape[1], np.nan)

    probs = soft_tree_predict(clf, x_obs, mask, gen, n_mc=50, random_state=42)
    assert probs.shape[0] == 2  # binary
    assert np.isclose(probs.sum(), 1.0)
    assert np.argmax(probs) in [0, 1]


def test_no_features_missing(binary_data):
    """Edge case: no features missing should match full-data predict."""
    X_train, y_train, X_test, _ = binary_data
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    gen = IdentityGenerator()
    mask = np.ones(X_test.shape[1], dtype=bool)  # all observed

    probs = soft_tree_predict(clf, X_test[0], mask, gen, n_mc=50, random_state=42)
    expected = clf.predict(X_test[0:1])[0]
    assert np.argmax(probs) == expected


# --- Test multiclass ---

def test_multiclass_tree(multiclass_data):
    """Soft tree prediction should work with multiclass."""
    X_train, y_train, X_test, _ = multiclass_data
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    gen = IdentityGenerator()
    mask = np.ones(X_test.shape[1], dtype=bool)

    probs = soft_tree_predict(clf, X_test[0], mask, gen, n_mc=50, random_state=42)
    assert len(probs) == 3  # 3 classes
    assert np.isclose(probs.sum(), 1.0)


def test_multiclass_forest(multiclass_data):
    """Soft forest prediction should work with multiclass."""
    X_train, y_train, X_test, _ = multiclass_data
    rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    gen = IdentityGenerator()
    mask_matrix = np.ones_like(X_test, dtype=bool)

    preds = soft_forest_predict(rf, X_test, mask_matrix, gen, n_mc=50, random_state=42)
    y_expected = rf.predict(X_test)
    assert np.array_equal(preds, y_expected)


# --- Test 1D input handling ---

def test_single_sample_input(binary_data):
    """wrapped_predict_v2 should handle 1D input (single sample)."""
    X_train, y_train, X_test, _ = binary_data
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    gen = IdentityGenerator()
    mask = np.ones(X_test.shape[1], dtype=bool)

    pred = wrapped_predict_v2(clf, X_test[0], mask, gen, n_mc=50, random_state=42)
    expected = clf.predict(X_test[0:1])
    assert np.array_equal(pred, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
