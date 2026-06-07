"""
Dataset distributional statistics for the paper dataset table.

The statistics are computed from the same preprocessed arrays used by the
experiments, after concatenating each dataset's train and test split.

Usage:
    python experiments/dataset_stats.py
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import skew

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data_loader import DATASETS as DATASET_LOADERS  # noqa: E402


DATASETS = [
    "breast_cancer",
    "banknote",
    "california",
    "letter",
    "adult",
    "credit",
]


def dataset_stats(name):
    X_train, y_train, X_test, y_test, _ = DATASET_LOADERS[name]()
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    df = pd.DataFrame(X)

    unique_counts = df.nunique(dropna=True)
    n_cont = int((unique_counts > 10).sum())
    n_binary = int((unique_counts == 2).sum())
    skews = df.apply(lambda col: skew(col, nan_policy="omit"))
    classes, counts = np.unique(y, return_counts=True)
    minority = counts.min() / counts.sum()

    return {
        "dataset": name,
        "n_features": X.shape[1],
        "n_continuous": n_cont,
        "pct_continuous": 100.0 * n_cont / X.shape[1],
        "n_binary": n_binary,
        "skew_min": float(np.nanmin(skews)),
        "skew_max": float(np.nanmax(skews)),
        "n_classes": len(classes),
        "minority_fraction": float(minority),
    }


def main():
    rows = [dataset_stats(name) for name in DATASETS]
    df = pd.DataFrame(rows)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results_v3")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "dataset_distribution_stats.csv")
    df.to_csv(csv_path, index=False)

    for row in rows:
        print(f"{row['dataset']}:")
        print(
            f"  Features: {row['n_features']} total, "
            f"{row['n_continuous']} continuous, {row['n_binary']} binary"
        )
        print(f"  Continuous features: {row['pct_continuous']:.1f}%")
        print(f"  Skewness range: [{row['skew_min']:.2f}, {row['skew_max']:.2f}]")
        print(f"  Minority class: {row['minority_fraction']:.1%}")
        print()

    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
