"""
Missingness mechanism implementations: MCAR, MAR, MNAR.
All functions return: (X_missing: np.ndarray with NaN, mask: np.ndarray bool True=observed)
"""
import numpy as np
from scipy.special import expit


def introduce_mcar(X, rate=0.3, random_state=None):
    """
    MCAR: each entry independently missing with probability `rate`.

    Args:
        X: np.ndarray of shape (n, d)
        rate: float, fraction of entries to mask
        random_state: int or None

    Returns:
        X_missing: np.ndarray with NaN for missing entries
        mask: np.ndarray bool, True=observed
    """
    rng = np.random.default_rng(random_state)
    mask = rng.random(X.shape) > rate  # True = observed
    X_missing = X.copy().astype(float)
    X_missing[~mask] = np.nan
    return X_missing, mask


def introduce_mcar_k(X, k, random_state=None):
    """
    MCAR: exactly k features missing per row.

    Args:
        X: np.ndarray of shape (n, d)
        k: int, number of features to drop per row
        random_state: int or None

    Returns:
        X_missing: np.ndarray with NaN for missing entries
        mask: np.ndarray bool, True=observed
    """
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    if k >= d:
        raise ValueError(f"k={k} must be < number of features d={d}")
    mask = np.ones((n, d), dtype=bool)
    cols = list(range(d))
    for i in range(n):
        missing_cols = rng.choice(cols, size=k, replace=False)
        mask[i, missing_cols] = False
    X_missing = X.copy().astype(float)
    X_missing[~mask] = np.nan
    return X_missing, mask


def introduce_mar(X, rate=0.3, random_state=None):
    """
    MAR: missingness of feature j depends on OTHER observed features.
    Pick 1/3 of features as 'drivers' (always observed).
    For each remaining feature, fit a logistic missingness model P(missing|drivers).

    Args:
        X: np.ndarray of shape (n, d)
        rate: float, target missingness rate
        random_state: int or None

    Returns:
        X_missing: np.ndarray with NaN
        mask: np.ndarray bool, True=observed
    """
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    mask = np.ones((n, d), dtype=bool)

    # Split features into drivers (always observed) and targets (can be missing)
    n_drivers = max(1, d // 3)
    driver_idx = rng.choice(d, size=n_drivers, replace=False)
    target_idx = np.setdiff1d(np.arange(d), driver_idx)

    # Standardize driver features for numerical stability
    X_drivers = X[:, driver_idx].copy()
    means = X_drivers.mean(axis=0)
    stds = X_drivers.std(axis=0) + 1e-8
    X_drivers = (X_drivers - means) / stds

    # Adjust per-target rate to achieve overall target rate
    # Only target features can be missing, so per-target rate must be higher
    n_target = len(target_idx)
    per_target_rate = rate * d / n_target

    for j in target_idx:
        # Generate missingness probability based on driver features
        w = rng.standard_normal(n_drivers)
        logits = X_drivers @ w
        # Calibrate to target rate using bias term
        target_r = min(per_target_rate, 0.95)  # cap at 95%
        bias = -np.quantile(logits, 1.0 - target_r)
        probs = expit(logits + bias)
        missing = rng.random(n) < probs
        mask[:, j] = ~missing

    X_missing = X.copy().astype(float)
    X_missing[~mask] = np.nan
    return X_missing, mask


def introduce_mnar(X, rate=0.3, random_state=None):
    """
    MNAR: missingness of feature j depends on the VALUE of feature j itself.
    Values far from the median are more likely to be missing (self-censoring).

    Args:
        X: np.ndarray of shape (n, d)
        rate: float, target missingness rate
        random_state: int or None

    Returns:
        X_missing: np.ndarray with NaN
        mask: np.ndarray bool, True=observed
    """
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    mask = np.ones((n, d), dtype=bool)

    for j in range(d):
        col = X[:, j]
        median_j = np.nanmedian(col)
        std_j = np.nanstd(col) + 1e-8
        # Probability increases with distance from median
        z = np.abs(col - median_j) / std_j

        # Calibrate alpha so average missing rate ~ target
        alpha = 1.0
        for _ in range(20):
            probs = expit(alpha * z - np.quantile(alpha * z, 1.0 - rate))
            actual_rate = probs.mean()
            if actual_rate > rate + 0.01:
                alpha *= 0.9
            elif actual_rate < rate - 0.01:
                alpha *= 1.1
            else:
                break

        probs = expit(alpha * z - np.quantile(alpha * z, 1.0 - rate))
        missing = rng.random(n) < probs
        mask[:, j] = ~missing

    X_missing = X.copy().astype(float)
    X_missing[~mask] = np.nan
    return X_missing, mask


# Convenience dispatcher
def apply_missingness(X, mechanism, rate, random_state=None):
    """
    Apply a missingness mechanism.

    Args:
        X: np.ndarray
        mechanism: str, one of 'mcar', 'mar', 'mnar'
        rate: float, target missingness rate
        random_state: int or None

    Returns:
        X_missing, mask
    """
    if mechanism == 'mcar':
        return introduce_mcar(X, rate=rate, random_state=random_state)
    elif mechanism == 'mar':
        return introduce_mar(X, rate=rate, random_state=random_state)
    elif mechanism == 'mnar':
        return introduce_mnar(X, rate=rate, random_state=random_state)
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")


if __name__ == '__main__':
    # Quick smoke test
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))

    for mech in ['mcar', 'mar', 'mnar']:
        for rate in [0.1, 0.3, 0.5]:
            X_miss, mask = apply_missingness(X, mech, rate, random_state=42)
            actual = 1.0 - mask.mean()
            print(f"{mech} rate={rate:.1f}  actual={actual:.3f}  "
                  f"NaN count={np.isnan(X_miss).sum()}")
