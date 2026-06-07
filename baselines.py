"""
All baseline methods for comparison with SurrogateML.
Each baseline takes standardized inputs and returns predictions.

Categories:
1. Impute-then-predict: impute missing values, then predict with frozen classifier
2. Native missing handling: models trained on the same clean split as SurrogateML,
   then evaluated with NaN only at test time (XGBoost, CatBoost, LightGBM)
3. Partial-distance KNN: compute distances using only observed features
4. Oracle: predict on complete (non-missing) data (upper bound)
"""
import numpy as np
from collections import Counter
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor


def _assert_clean_training_split(X_train, baseline_name):
    """
    Native-missingness baselines must share SurrogateML's clean-training protocol.

    Missingness is introduced only in the test split. This prevents the native
    baselines from being trained under a different data distribution than the
    frozen model wrapped by SurrogateML.
    """
    X_arr = np.asarray(X_train, dtype=float)
    if np.isnan(X_arr).any():
        raise ValueError(
            f"{baseline_name} received NaNs in X_train. Native-missingness "
            "baselines must be trained on the clean training split; introduce "
            "missingness only in X_test."
        )


# ============================================================
# Impute-then-predict baselines
# ============================================================

def baseline_mean_impute(clf, X_train, X_test_missing, mask_matrix=None):
    """Mean imputation on missing features, then predict with frozen clf."""
    imp = SimpleImputer(strategy='mean').fit(X_train)
    X_imputed = imp.transform(X_test_missing)
    return clf.predict(X_imputed)


def baseline_knn_impute(clf, X_train, X_test_missing, mask_matrix=None, n_neighbors=5):
    """KNN imputation, then predict."""
    imp = KNNImputer(n_neighbors=n_neighbors).fit(X_train)
    X_imputed = imp.transform(X_test_missing)
    return clf.predict(X_imputed)


def baseline_mice_impute(clf, X_train, X_test_missing, mask_matrix=None):
    """MICE (IterativeImputer), then predict."""
    imp = IterativeImputer(random_state=42, max_iter=10).fit(X_train)
    X_imputed = imp.transform(X_test_missing)
    return clf.predict(X_imputed)


def make_rf_imputer(n_estimators=50, max_iter=10, random_state=42):
    """
    sklearn IterativeImputer with a RandomForestRegressor estimator.

    Conceptually similar to MissForest, but exposed as an explicit sklearn
    RF-based imputation baseline for reviewer-facing tables.
    """
    return IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        ),
        initial_strategy='mean',
        max_iter=max_iter,
        random_state=random_state,
    )


def baseline_rf_impute(clf, X_train, X_test_missing, mask_matrix=None):
    """Explicit sklearn RF-Imp baseline, then predict with frozen clf."""
    imp = make_rf_imputer().fit(X_train)
    X_imputed = imp.transform(X_test_missing)
    return clf.predict(X_imputed)


def _fit_rf_tree_imputer(X_train):
    """Fit a MissForest-style iterative Random Forest imputer on clean training data."""
    return IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        initial_strategy='mean',
        random_state=42,
        max_iter=10,
    ).fit(X_train)


def baseline_rf_tree_impute(clf, X_train, X_test_missing, mask_matrix=None):
    """Tree-based RF imputation (MissForest-style), then predict with frozen clf."""
    imp = _fit_rf_tree_imputer(X_train)
    X_imputed = imp.transform(X_test_missing)
    return clf.predict(X_imputed)


def baseline_missforest_impute(clf, X_train, X_test_missing, mask_matrix=None):
    """Backward-compatible alias for RF tree imputation."""
    return baseline_rf_tree_impute(clf, X_train, X_test_missing, mask_matrix)


# ============================================================
# Models with native missing handling
# ============================================================

def baseline_xgboost_native(X_train, y_train, X_test_missing):
    """XGBoost trained on clean data, tested with NaN (native handling)."""
    _assert_clean_training_split(X_train, "xgboost_native")
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost not installed. Run: pip install xgboost")

    n_classes = len(np.unique(y_train))
    params = dict(
        n_estimators=100, max_depth=5, random_state=42,
        eval_metric='logloss', tree_method='hist'
    )
    if n_classes > 2:
        params['objective'] = 'multi:softmax'
        params['num_class'] = n_classes

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model.predict(X_test_missing)


def baseline_catboost_native(X_train, y_train, X_test_missing):
    """CatBoost trained on clean data, tested with NaN (native handling)."""
    _assert_clean_training_split(X_train, "catboost_native")
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        raise ImportError("catboost not installed. Run: pip install catboost")

    model = CatBoostClassifier(
        iterations=100, depth=5, random_seed=42, verbose=0
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test_missing)
    return preds.flatten().astype(int)


def baseline_lightgbm_native(X_train, y_train, X_test_missing):
    """LightGBM trained on clean data, tested with NaN (native handling)."""
    _assert_clean_training_split(X_train, "lightgbm_native")
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm not installed. Run: pip install lightgbm")

    model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=5, random_state=42, verbose=-1
    )
    model.fit(X_train, y_train)
    return model.predict(X_test_missing)


# ============================================================
# Partial-distance KNN (no imputation)
# ============================================================

def baseline_partial_distance_knn(X_train, y_train, X_test_missing, mask_matrix, k=5):
    """
    KNN using only observed features for distance computation.
    For each test sample, compute distance to all training samples using
    only the features that are observed in the test sample.
    """
    n_test = X_test_missing.shape[0]
    results = []

    for i in range(n_test):
        obs_idx = mask_matrix[i]  # which features are observed
        if obs_idx.sum() == 0:
            # No features observed -> predict majority class
            results.append(Counter(y_train).most_common(1)[0][0])
            continue

        # Compute distances using only observed features
        x_obs = X_test_missing[i, obs_idx]
        X_tr_obs = X_train[:, obs_idx]
        dists = np.sqrt(((X_tr_obs - x_obs) ** 2).sum(axis=1))

        # k nearest neighbors
        nn_idx = np.argsort(dists)[:k]
        nn_labels = y_train[nn_idx]

        # Majority vote
        results.append(Counter(nn_labels).most_common(1)[0][0])

    return np.array(results)


# ============================================================
# Oracle (upper bound)
# ============================================================

def baseline_oracle(clf, X_test_full):
    """Upper bound: predict on complete test data (no missingness)."""
    return clf.predict(X_test_full)


# ============================================================
# Registries
# ============================================================

IMPUTE_BASELINES = {
    'mean_impute': baseline_mean_impute,
    'knn_impute': baseline_knn_impute,
    'mice_impute': baseline_mice_impute,
    'rf_impute': baseline_rf_impute,
    'rf_tree_impute': baseline_rf_tree_impute,
    'missforest_impute': baseline_missforest_impute,
}

NATIVE_BASELINES = {
    'xgboost_native': baseline_xgboost_native,
    'catboost_native': baseline_catboost_native,
    'lightgbm_native': baseline_lightgbm_native,
}

OTHER_BASELINES = {
    'partial_knn': baseline_partial_distance_knn,
}
