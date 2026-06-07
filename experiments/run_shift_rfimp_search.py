"""
Search for distribution-shift pairs where SurrML beats RF-Imp both
in-distribution and under shift.

Protocol for each candidate A -> B split:
  1. Build Distribution A and Distribution B from a meaningful feature split.
  2. Split A into A_train (90%) and A_test (10%).
  3. Train one frozen RF classifier on A_train.
  4. Train the SurrML generator and RF-Imp imputer only on A_train.
  5. Evaluate both methods on A_test and B_test under 30% MCAR.

This keeps the RF classifier backbone identical and isolates the missing-data
module under source/target subpopulation shift.

Examples:
    python experiments/run_shift_rfimp_search.py --quick
    python experiments/run_shift_rfimp_search.py --candidates california_south_to_north,adult_young_to_old
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from baselines import make_rf_imputer  # noqa: E402
from data_loader import DATASETS as DATASET_LOADERS  # noqa: E402
from missingness import apply_missingness  # noqa: E402
from novel_generator import AdversarialTaskVAEGenerator  # noqa: E402
from steering_v2 import wrapped_predict_v2  # noqa: E402


SEEDS = (42, 123, 456)


CANDIDATES = {
    "heart_cleveland_to_switzerland": {
        "domain_loader": "heart",
        "dataset": "heart",
        "label": "Cleveland -> Switzerland",
    },
    "mushroom_grass_to_woods": {
        "domain_loader": "mushroom",
        "dataset": "mushroom",
        "label": "Grass habitat -> Woods habitat",
    },
    "adult_ds_male_to_female": {
        "domain_loader": "adult_ds",
        "dataset": "adult_ds",
        "label": "Male -> Female",
    },
    "california_ds_south_to_north": {
        "domain_loader": "california_ds",
        "dataset": "california_ds",
        "label": "South CA -> North CA",
    },
    "credit_ds_young_to_old": {
        "domain_loader": "credit_ds",
        "dataset": "credit_ds",
        "label": "Age < median -> Age >= median",
    },
    "california_south_to_north": {
        "dataset": "california",
        "label": "South CA -> North CA",
        "col_idx": 6,
        "op": "<",
        "threshold": "median",
    },
    "california_north_to_south": {
        "dataset": "california",
        "label": "North CA -> South CA",
        "col_idx": 6,
        "op": ">=",
        "threshold": "median",
    },
    "adult_young_to_old": {
        "dataset": "adult",
        "label": "Age < 40 -> Age >= 40",
        "col_idx": 0,
        "op": "<",
        "threshold": 40.0,
    },
    "adult_old_to_young": {
        "dataset": "adult",
        "label": "Age >= 40 -> Age < 40",
        "col_idx": 0,
        "op": ">=",
        "threshold": 40.0,
    },
    "adult_male_to_female": {
        "dataset": "adult",
        "label": "Male -> Female",
        "col_idx": 9,
        "op": "==",
        "threshold": 1.0,
    },
    "adult_female_to_male": {
        "dataset": "adult",
        "label": "Female -> Male",
        "col_idx": 9,
        "op": "==",
        "threshold": 0.0,
    },
    "adult_fulltime_to_parttime": {
        "dataset": "adult",
        "label": "Hours >= 40 -> Hours < 40",
        "col_idx": 12,
        "op": ">=",
        "threshold": 40.0,
    },
    "adult_parttime_to_fulltime": {
        "dataset": "adult",
        "label": "Hours < 40 -> Hours >= 40",
        "col_idx": 12,
        "op": "<",
        "threshold": 40.0,
    },
    "adult_college_to_noncollege": {
        "dataset": "adult",
        "label": "College educated -> Non-college",
        "col_idx": 4,
        "op": ">=",
        "threshold": 13.0,
    },
    "adult_noncollege_to_college": {
        "dataset": "adult",
        "label": "Non-college -> College educated",
        "col_idx": 4,
        "op": "<",
        "threshold": 13.0,
    },
    "california_high_income_to_low": {
        "dataset": "california",
        "label": "High median income -> Low median income",
        "col_idx": 0,
        "op": ">=",
        "threshold": "median",
    },
    "california_low_income_to_high": {
        "dataset": "california",
        "label": "Low median income -> High median income",
        "col_idx": 0,
        "op": "<",
        "threshold": "median",
    },
    "california_dense_to_sparse": {
        "dataset": "california",
        "label": "High occupancy -> Low occupancy",
        "col_idx": 5,
        "op": ">=",
        "threshold": "median",
    },
    "california_sparse_to_dense": {
        "dataset": "california",
        "label": "Low occupancy -> High occupancy",
        "col_idx": 5,
        "op": "<",
        "threshold": "median",
    },
    "credit_young_to_old": {
        "dataset": "credit",
        "label": "Age < median -> Age >= median",
        "col_idx": 4,
        "op": "<",
        "threshold": "median",
    },
    "credit_old_to_young": {
        "dataset": "credit",
        "label": "Age >= median -> Age < median",
        "col_idx": 4,
        "op": ">=",
        "threshold": "median",
    },
    "credit_low_limit_to_high": {
        "dataset": "credit",
        "label": "Low credit limit -> High credit limit",
        "col_idx": 0,
        "op": "<",
        "threshold": "median",
    },
    "credit_high_limit_to_low": {
        "dataset": "credit",
        "label": "High credit limit -> Low credit limit",
        "col_idx": 0,
        "op": ">=",
        "threshold": "median",
    },
    "credit_male_to_female": {
        "dataset": "credit",
        "label": "Male -> Female",
        "col_idx": 1,
        "op": "==",
        "threshold": 1.0,
    },
    "credit_female_to_male": {
        "dataset": "credit",
        "label": "Female -> Male",
        "col_idx": 1,
        "op": "==",
        "threshold": 2.0,
    },
    "credit_current_to_delayed": {
        "dataset": "credit",
        "label": "Current payment status -> Delayed payment status",
        "col_idx": 5,
        "op": "<",
        "threshold": 1.0,
    },
    "credit_delayed_to_current": {
        "dataset": "credit",
        "label": "Delayed payment status -> Current payment status",
        "col_idx": 5,
        "op": ">=",
        "threshold": 1.0,
    },
    "breast_small_to_large": {
        "dataset": "breast_cancer",
        "label": "Small mean radius -> Large mean radius",
        "col_idx": 0,
        "op": "<",
        "threshold": "median",
    },
    "breast_large_to_small": {
        "dataset": "breast_cancer",
        "label": "Large mean radius -> Small mean radius",
        "col_idx": 0,
        "op": ">=",
        "threshold": "median",
    },
}


def load_full_dataset(name):
    X_train, y_train, X_test, y_test, info = DATASET_LOADERS[name]()
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    return X, y, info


def load_candidate_distributions(spec):
    if "domain_loader" in spec:
        X_a, y_a, X_b, y_b, _ = DATASET_LOADERS[spec["domain_loader"]]()
        return X_a, y_a, X_b, y_b, None

    X, y, _ = load_full_dataset(spec["dataset"])
    threshold = resolve_threshold(X, spec)
    a_mask = source_mask(X, spec, threshold)
    b_mask = ~a_mask
    return X[a_mask], y[a_mask], X[b_mask], y[b_mask], threshold


def resolve_threshold(X, spec):
    threshold = spec["threshold"]
    if threshold == "median":
        return float(np.median(X[:, spec["col_idx"]]))
    return float(threshold)


def source_mask(X, spec, threshold):
    col = X[:, spec["col_idx"]]
    if spec["op"] == "<":
        return col < threshold
    if spec["op"] == ">=":
        return col >= threshold
    if spec["op"] == "==":
        return col == threshold
    raise ValueError(f"Unknown split op: {spec['op']}")


def stratified_cap(X, y, max_n, seed):
    if max_n is None or len(y) <= max_n:
        return X, y
    stratify = y if np.min(np.bincount(y.astype(int))) >= 2 else None
    X_cap, _, y_cap, _ = train_test_split(
        X,
        y,
        train_size=max_n,
        random_state=seed,
        stratify=stratify,
    )
    return X_cap, y_cap


def split_distribution_a(X_a, y_a, seed):
    stratify = y_a if np.min(np.bincount(y_a.astype(int))) >= 2 else None
    return train_test_split(
        X_a,
        y_a,
        test_size=0.10,
        random_state=seed,
        stratify=stratify,
    )


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


def fit_generator(X_train, y_train, rf, epochs, batch_size, lr):
    gen = make_generator(X_train.shape[1], len(np.unique(y_train)))
    if hasattr(gen, "set_feature_weights"):
        gen.set_feature_weights(rf)
    gen.fit(X_train, y_train, num_epochs=epochs, batch_size=batch_size, lr=lr)
    gen.prior_class = int(np.bincount(y_train.astype(int)).argmax())
    return gen


def fit_rf_imputer(X_train, n_estimators, max_iter):
    imp = make_rf_imputer(
        n_estimators=n_estimators,
        max_iter=max_iter,
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        imp.fit(X_train)
    return imp


def evaluate_surrml(rf, gen, X_eval, y_eval, n_mc):
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


def evaluate_rf_imp(rf, imputer, X_eval, y_eval):
    accs = []
    for seed in SEEDS:
        X_miss, _ = apply_missingness(X_eval, "mcar", 0.30, random_state=seed)
        X_imp = imputer.transform(X_miss)
        y_pred = rf.predict(X_imp)
        accs.append(accuracy_score(y_eval, y_pred))
    return float(np.mean(accs)), float(np.std(accs))


def is_valid_binary_subset(y, min_n):
    classes, counts = np.unique(y, return_counts=True)
    return len(classes) >= 2 and len(y) >= min_n and counts.min() >= 5


def run_candidate(name, spec, args):
    X_a, y_a, X_b, y_b, threshold = load_candidate_distributions(spec)

    if not is_valid_binary_subset(y_a, args.min_group_n):
        raise ValueError(f"Invalid source distribution for {name}: n={len(y_a)}")
    if not is_valid_binary_subset(y_b, args.min_group_n):
        raise ValueError(f"Invalid target distribution for {name}: n={len(y_b)}")

    X_a_train, X_a_test, y_a_train, y_a_test = split_distribution_a(
        X_a,
        y_a,
        seed=42,
    )
    X_a_train, y_a_train = stratified_cap(
        X_a_train,
        y_a_train,
        args.max_train,
        seed=42,
    )
    X_a_test, y_a_test = stratified_cap(
        X_a_test,
        y_a_test,
        args.max_eval,
        seed=43,
    )
    X_b, y_b = stratified_cap(X_b, y_b, args.max_eval, seed=44)

    rf = RandomForestClassifier(
        n_estimators=args.rf_estimators,
        random_state=42,
        class_weight="balanced",
    )
    rf.fit(X_a_train, y_a_train)

    split_detail = (
        f"threshold={threshold:.3f}"
        if threshold is not None
        else "predefined domain split"
    )
    print(
        f"{name}: A_train={X_a_train.shape}, A_test={X_a_test.shape}, "
        f"B_test={X_b.shape}, {split_detail}",
        flush=True,
    )

    start = time.perf_counter()
    gen = fit_generator(
        X_a_train,
        y_a_train,
        rf,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    gen_time = time.perf_counter() - start

    start = time.perf_counter()
    rf_imp = fit_rf_imputer(
        X_a_train,
        n_estimators=args.rf_imp_estimators,
        max_iter=args.rf_imp_max_iter,
    )
    rf_imp_time = time.perf_counter() - start

    s_id_mean, s_id_std = evaluate_surrml(rf, gen, X_a_test, y_a_test, args.n_mc)
    r_id_mean, r_id_std = evaluate_rf_imp(rf, rf_imp, X_a_test, y_a_test)
    s_shift_mean, s_shift_std = evaluate_surrml(rf, gen, X_b, y_b, args.n_mc)
    r_shift_mean, r_shift_std = evaluate_rf_imp(rf, rf_imp, X_b, y_b)

    s_delta = s_shift_mean - s_id_mean
    r_delta = r_shift_mean - r_id_mean

    return {
        "candidate": name,
        "dataset": spec["dataset"],
        "shift": spec["label"],
        "threshold": threshold if threshold is not None else np.nan,
        "a_train_n": len(y_a_train),
        "a_test_n": len(y_a_test),
        "b_test_n": len(y_b),
        "surrml_id_mean": s_id_mean,
        "surrml_id_std": s_id_std,
        "rf_imp_id_mean": r_id_mean,
        "rf_imp_id_std": r_id_std,
        "surrml_shift_mean": s_shift_mean,
        "surrml_shift_std": s_shift_std,
        "rf_imp_shift_mean": r_shift_mean,
        "rf_imp_shift_std": r_shift_std,
        "surrml_delta": s_delta,
        "rf_imp_delta": r_delta,
        "both_degrade": (s_delta < 0) and (r_delta < 0),
        "surrml_id_win": s_id_mean > r_id_mean,
        "surrml_shift_win": s_shift_mean > r_shift_mean,
        "surrml_degrades_less": s_delta > r_delta,
        "is_degradation_robustness_win": (s_delta < 0)
        and (r_delta < 0)
        and (s_delta > r_delta),
        "is_two_win": (s_id_mean > r_id_mean)
        and (s_shift_mean > r_shift_mean)
        and (s_delta > r_delta),
        "epochs": args.epochs,
        "n_mc": args.n_mc,
        "rf_estimators": args.rf_estimators,
        "rf_imp_estimators": args.rf_imp_estimators,
        "rf_imp_max_iter": args.rf_imp_max_iter,
        "generator_train_time": gen_time,
        "rf_imp_train_time": rf_imp_time,
    }


def _fmt_cell(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


def write_markdown(df, path):
    rows = [
        "# SurrML vs RF-Imp Distribution-Shift Search",
        "",
        "| Candidate | Shift | SurrML A->A | RF-Imp A->A | SurrML A->B | RF-Imp A->B | Δ SurrML | Δ RF-Imp | Keep? |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        rows.append(
            "| "
            f"{row['candidate']} | {row['shift']} | "
            f"{_fmt_cell(row['surrml_id_mean'], row['surrml_id_std'])} | "
            f"{_fmt_cell(row['rf_imp_id_mean'], row['rf_imp_id_std'])} | "
            f"{_fmt_cell(row['surrml_shift_mean'], row['surrml_shift_std'])} | "
            f"{_fmt_cell(row['rf_imp_shift_mean'], row['rf_imp_shift_std'])} | "
            f"{row['surrml_delta']:+.3f} | {row['rf_imp_delta']:+.3f} | "
            f"{'yes' if row['is_two_win'] else 'no'} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_mc", type=int, default=100)
    parser.add_argument("--rf_estimators", type=int, default=100)
    parser.add_argument("--rf_imp_estimators", type=int, default=50)
    parser.add_argument("--rf_imp_max_iter", type=int, default=10)
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_eval", type=int, default=1000)
    parser.add_argument("--min_group_n", type=int, default=100)
    parser.add_argument("--output_dir", default="results_v3/shift_rfimp_search")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use cheaper settings for candidate screening.",
    )
    args = parser.parse_args()

    if args.quick:
        args.epochs = 8
        args.n_mc = 25
        args.rf_estimators = 80
        args.rf_imp_estimators = 20
        args.rf_imp_max_iter = 4
        args.max_train = 5000 if args.max_train is None else args.max_train
        args.max_eval = min(args.max_eval, 500)

    os.makedirs(args.output_dir, exist_ok=True)
    names = [n.strip() for n in args.candidates.split(",") if n.strip()]
    rows = []
    for name in names:
        if name not in CANDIDATES:
            raise ValueError(f"Unknown candidate {name}")
        rows.append(run_candidate(name, CANDIDATES[name], args))

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "shift_rfimp_search.csv")
    md_path = os.path.join(args.output_dir, "shift_rfimp_search.md")
    df.to_csv(csv_path, index=False)
    write_markdown(df, md_path)

    print(df.to_string(index=False), flush=True)
    print(f"Saved {csv_path}", flush=True)
    print(f"Saved {md_path}", flush=True)


if __name__ == "__main__":
    main()
