import numpy as np
import pytest

from baselines import _assert_clean_training_split
from run_experiment import TRAINING_PROTOCOL, run_single_experiment


def test_native_baseline_rejects_corrupted_training_split():
    X_train = np.array([[0.0, 1.0], [np.nan, 2.0]])

    with pytest.raises(ValueError, match="clean training split"):
        _assert_clean_training_split(X_train, "xgboost_native")


def test_run_single_experiment_records_test_only_missingness_protocol(monkeypatch):
    import run_experiment as runner

    X_train = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
    ])
    y_test = np.array([0, 1])
    info = {"name": "tiny", "n_features": 2, "n_classes": 2}

    def fake_apply_missingness(X, mechanism, rate, random_state=None):
        X_missing = X.copy().astype(float)
        mask = np.ones_like(X_missing, dtype=bool)
        mask[:, 1] = False
        X_missing[~mask] = np.nan
        return X_missing, mask

    class FakeGenerator:
        prior_class = 0

        def sample(self, x_obs, mask, n_samples, rng):
            x = np.nan_to_num(x_obs, nan=0.0)
            return np.tile(x, (n_samples, 1))

    monkeypatch.setitem(runner.CLASSIFIERS, "TinyDT", ("DecisionTreeClassifier", {"random_state": 0}))
    monkeypatch.setitem(runner.MISSINGNESS, "tiny_mcar", ("mcar", 0.5))
    monkeypatch.setattr(runner, "apply_missingness", fake_apply_missingness)
    monkeypatch.setattr(runner, "IMPUTE_BASELINES", {})
    monkeypatch.setattr(runner, "NATIVE_BASELINES", {})
    monkeypatch.setattr(runner, "OTHER_BASELINES", {})
    monkeypatch.setattr(runner, "wrapped_predict_v2", lambda *args, **kwargs: np.array([0, 1]))

    results = run_single_experiment(
        "tiny",
        "TinyDT",
        "tiny_mcar",
        "atvae",
        run_id=0,
        preloaded_data=(X_train, y_train, X_test, y_test, info),
        pretrained_gen=FakeGenerator(),
    )

    assert results
    for row in results:
        assert row["training_protocol"] == TRAINING_PROTOCOL
        assert row["train_missing_rate"] == 0.0
        assert row["test_missing_rate_actual"] == 0.5
