"""
All-dataset simulated distribution-shift experiment.

For each selected benchmark dataset:
  1. Train the frozen RF on the full clean training split.
  2. Train an in-distribution HetIWAE generator on the full training split.
  3. Train a shifted HetIWAE generator on one source subpopulation.
  4. Fit RF-Imp on the same full and shifted-source training sets.
  5. Evaluate both missing-data modules on the same target subpopulation
     under 30% MCAR.

This isolates missing-data-module robustness while keeping the downstream RF fixed.

Usage:
    python experiments/run_shift_all.py --epochs 50
    python experiments/run_shift_all.py --datasets california,adult --epochs 30
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from baselines import make_rf_imputer  # noqa: E402
from data_loader import DATASETS as DATASET_LOADERS  # noqa: E402
from missingness import apply_missingness  # noqa: E402
from novel_generator import AdversarialTaskVAEGenerator  # noqa: E402
from steering_v2 import wrapped_predict_v2  # noqa: E402


DATASET_SPLITS = {
    "breast_cancer": {
        "label": "Mean radius < median -> >= median",
        "col_idx": 0,
        "threshold": "median",
    },
    "banknote": {
        "label": "Variance < median -> >= median",
        "col_idx": 0,
        "threshold": "median",
    },
    "california": {
        "label": "South CA -> North CA",
        "col_idx": 6,  # latitude
        "threshold": "median",
    },
    "letter": {
        "label": "x-box < median -> >= median",
        "col_idx": 0,
        "threshold": "median",
    },
    "adult": {
        "label": "Age < 40 -> >= 40",
        "col_idx": 0,
        "threshold": 40.0,
    },
    "credit": {
        "label": "Age < median -> >= median",
        "col_idx": 4,  # AGE after dropping ID and target
        "threshold": "median",
    },
}
SEEDS = (42, 123, 456)


def resolve_threshold(X, spec):
    threshold = spec["threshold"]
    if threshold == "median":
        return float(np.median(X[:, spec["col_idx"]]))
    return float(threshold)


def subpopulation_split(X, y, spec):
    threshold = resolve_threshold(X, spec)
    col = X[:, spec["col_idx"]]
    train_mask = col < threshold
    eval_mask = col >= threshold
    if train_mask.sum() == 0 or eval_mask.sum() == 0:
        raise ValueError(f"Empty split for threshold {threshold}")
    return X[train_mask], y[train_mask], X[eval_mask], y[eval_mask], threshold


def make_generator(input_dim, n_classes):
    return AdversarialTaskVAEGenerator(
        input_dim=input_dim,
        n_classes=n_classes,
        latent_dim=64,
        hidden_dims=(256, 256),
        k_iw=20,
        device="cpu",
        lambda_adv=0.0,
        n_critic=0,
    )


def fit_generator(X, y, rf, n_classes, epochs, batch_size, lr):
    gen = make_generator(X.shape[1], n_classes)
    if hasattr(gen, "set_feature_weights"):
        gen.set_feature_weights(rf)
    gen.fit(X, y, num_epochs=epochs, batch_size=batch_size, lr=lr)
    gen.prior_class = int(np.bincount(y).argmax())
    return gen


def evaluate_generator(rf, gen, X_eval, y_eval, n_mc):
    accs = []
    for seed in SEEDS:
        X_miss, mask = apply_missingness(X_eval, "mcar", 0.30, random_state=seed)
        y_pred = wrapped_predict_v2(
            rf,
            X_miss,
            mask,
            gen,
            n_mc=n_mc,
            random_state=seed,
        )
        accs.append(accuracy_score(y_eval, y_pred))
    return float(np.mean(accs)), float(np.std(accs))


def fit_rf_imputer(X_train, n_estimators, max_iter, random_state):
    imp = make_rf_imputer(
        n_estimators=n_estimators,
        max_iter=max_iter,
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        imp.fit(X_train)
    return imp


def evaluate_rf_imputer(rf, imputer, X_eval, y_eval):
    accs = []
    for seed in SEEDS:
        X_miss, _ = apply_missingness(X_eval, "mcar", 0.30, random_state=seed)
        X_imp = imputer.transform(X_miss)
        y_pred = rf.predict(X_imp)
        accs.append(accuracy_score(y_eval, y_pred))
    return float(np.mean(accs)), float(np.std(accs))


def cap_eval_split(X_eval, y_eval, max_eval, seed=42):
    if max_eval is None or len(y_eval) <= max_eval:
        return X_eval, y_eval
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_eval), size=max_eval, replace=False)
    idx.sort()
    return X_eval[idx], y_eval[idx]


def run_dataset(
    dataset,
    epochs,
    batch_size,
    lr,
    n_mc,
    max_eval,
    rf_imp_estimators,
    rf_imp_max_iter,
):
    spec = DATASET_SPLITS[dataset]
    X_train, y_train, _, _, _ = DATASET_LOADERS[dataset]()
    X_shift_train, y_shift_train, X_eval, y_eval, threshold = subpopulation_split(
        X_train,
        y_train,
        spec,
    )
    X_eval, y_eval = cap_eval_split(X_eval, y_eval, max_eval)

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    n_classes = len(np.unique(y_train))

    print(
        f"{dataset}: shift_train={X_shift_train.shape}, eval={X_eval.shape}, "
        f"threshold={threshold:.3f}",
        flush=True,
    )

    start = time.perf_counter()
    print(f"  training full generator ({epochs} epochs)", flush=True)
    gen_full = fit_generator(X_train, y_train, rf, n_classes, epochs, batch_size, lr)
    full_train_time = time.perf_counter() - start

    start = time.perf_counter()
    print(f"  training shifted generator ({epochs} epochs)", flush=True)
    gen_shift = fit_generator(
        X_shift_train,
        y_shift_train,
        rf,
        n_classes,
        epochs,
        batch_size,
        lr,
    )
    shift_train_time = time.perf_counter() - start

    base_mean, base_std = evaluate_generator(rf, gen_full, X_eval, y_eval, n_mc)
    shift_mean, shift_std = evaluate_generator(rf, gen_shift, X_eval, y_eval, n_mc)

    start = time.perf_counter()
    print("  fitting RF-Imp on full training split", flush=True)
    rf_imp_full = fit_rf_imputer(
        X_train,
        n_estimators=rf_imp_estimators,
        max_iter=rf_imp_max_iter,
        random_state=42,
    )
    rf_imp_full_train_time = time.perf_counter() - start

    start = time.perf_counter()
    print("  fitting RF-Imp on shifted source subpopulation", flush=True)
    rf_imp_shift = fit_rf_imputer(
        X_shift_train,
        n_estimators=rf_imp_estimators,
        max_iter=rf_imp_max_iter,
        random_state=42,
    )
    rf_imp_shift_train_time = time.perf_counter() - start

    rf_imp_base_mean, rf_imp_base_std = evaluate_rf_imputer(
        rf,
        rf_imp_full,
        X_eval,
        y_eval,
    )
    rf_imp_shift_mean, rf_imp_shift_std = evaluate_rf_imputer(
        rf,
        rf_imp_shift,
        X_eval,
        y_eval,
    )

    return {
        "dataset": dataset,
        "split": spec["label"],
        "threshold": threshold,
        "shift_train_n": len(y_shift_train),
        "eval_n": len(y_eval),
        "max_eval": max_eval,
        "in_distribution_mean": base_mean,
        "in_distribution_std": base_std,
        "shifted_mean": shift_mean,
        "shifted_std": shift_std,
        "delta_shift_minus_base": shift_mean - base_mean,
        "rf_imp_in_distribution_mean": rf_imp_base_mean,
        "rf_imp_in_distribution_std": rf_imp_base_std,
        "rf_imp_shifted_mean": rf_imp_shift_mean,
        "rf_imp_shifted_std": rf_imp_shift_std,
        "rf_imp_delta_shift_minus_base": rf_imp_shift_mean - rf_imp_base_mean,
        "epochs": epochs,
        "n_mc": n_mc,
        "rf_imp_estimators": rf_imp_estimators,
        "rf_imp_max_iter": rf_imp_max_iter,
        "full_generator_train_time": full_train_time,
        "shifted_generator_train_time": shift_train_time,
        "rf_imp_full_train_time": rf_imp_full_train_time,
        "rf_imp_shifted_train_time": rf_imp_shift_train_time,
    }


def _cell(mean, std):
    return f"{mean:.3f}\\scriptsize{{$\\pm${std:.3f}}}"


def _latex_shift_label(label):
    return (
        label
        .replace("->", "$\\rightarrow$")
        .replace(">= median", "$\\geq$ median")
        .replace(">= 40", "$\\geq$ 40")
        .replace("< median", "$<$ median")
        .replace("< 40", "$<$ 40")
    )


def write_latex(df, path):
    lines = [
        "\\begin{table*}[t]",
        "\\caption{Robustness under simulated deployment shift on selected benchmark datasets. SurrML and RF-Imp both wrap the same frozen RF under 30\\% MCAR missingness. In-distribution missing-data modules are trained on the full clean training split; shifted modules are trained on one source subpopulation and evaluated on another. Values are mean\\,$\\pm$\\,std over three missingness seeds.}",
        "\\label{tab:distribution_shift}",
        "\\centering\\scriptsize",
        "\\begin{tabular}{llrcccccc}",
        "\\toprule",
        "Dataset & Shift & Eval $n$ & SurrML full & SurrML shift & $\\Delta_S$ & RF-Imp full & RF-Imp shift & $\\Delta_R$ \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['dataset'].replace('_', ' ').title()} & {_latex_shift_label(row['split'])} & "
            f"{int(row['eval_n'])} & "
            f"{_cell(row['in_distribution_mean'], row['in_distribution_std'])} & "
            f"{_cell(row['shifted_mean'], row['shifted_std'])} & "
            f"{row['delta_shift_minus_base']:+.3f} & "
            f"{_cell(row['rf_imp_in_distribution_mean'], row['rf_imp_in_distribution_std'])} & "
            f"{_cell(row['rf_imp_shifted_mean'], row['rf_imp_shifted_std'])} & "
            f"{row['rf_imp_delta_shift_minus_base']:+.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(DATASET_SPLITS))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_mc", type=int, default=100)
    parser.add_argument("--max_eval", type=int, default=1000)
    parser.add_argument("--rf_imp_estimators", type=int, default=50)
    parser.add_argument("--rf_imp_max_iter", type=int, default=10)
    parser.add_argument("--output_dir", default="results_v3")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    rows = []
    for dataset in datasets:
        rows.append(
            run_dataset(
                dataset,
                args.epochs,
                args.batch_size,
                args.lr,
                args.n_mc,
                args.max_eval,
                args.rf_imp_estimators,
                args.rf_imp_max_iter,
            )
        )

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "distribution_shift_all_summary.csv")
    tex_path = os.path.join(args.output_dir, "distribution_shift_all_table.tex")
    df.to_csv(csv_path, index=False)
    write_latex(df, tex_path)
    print(df.to_string(index=False), flush=True)
    print(f"Saved {csv_path}", flush=True)
    print(f"Saved {tex_path}", flush=True)


if __name__ == "__main__":
    main()
