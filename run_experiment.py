"""
Unified experiment runner for SurrogateML.

Usage:
    python run_experiment.py --dataset heart --classifier DT --missingness mcar_30 --n_runs 10
    python run_experiment.py --dataset heart --classifier all --missingness all --n_runs 3
    python run_experiment.py --dataset all --classifier all --missingness all --n_runs 10
"""
import argparse
import time
import os
import sys
import json
import traceback
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score)
from sklearn.impute import SimpleImputer

from data_loader import DATASETS as DATASET_LOADERS
from missingness import apply_missingness
from baselines import (IMPUTE_BASELINES, NATIVE_BASELINES, OTHER_BASELINES,
                       baseline_oracle)
from steering_v2 import wrapped_predict_v2
from experiment_config import *


TRAINING_PROTOCOL = "clean_train__test_missing_only"


def assert_clean_training_split(X_train, context):
    """Fail fast if an experiment tries to train any model on corrupted data."""
    if np.isnan(np.asarray(X_train, dtype=float)).any():
        raise ValueError(
            f"{context} contains NaNs. All models in the unified protocol must "
            "train on the same clean split; missingness is introduced only in "
            "the test split."
        )


def get_classifier(name):
    """Instantiate a classifier by name."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    cls_name, params = CLASSIFIERS[name]
    cls_map = {
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'RandomForestClassifier': RandomForestClassifier,
        'KNeighborsClassifier': KNeighborsClassifier,
        'LogisticRegression': LogisticRegression,
        'MLPClassifier': MLPClassifier,
    }
    return cls_map[cls_name](**params)


def get_generator(gen_type, input_dim, n_classes, device='cpu'):
    """Instantiate a generator by type."""
    if gen_type == 'atvae':
        from novel_generator import AdversarialTaskVAEGenerator
        return AdversarialTaskVAEGenerator(
            input_dim=input_dim, n_classes=n_classes,
            latent_dim=64, hidden_dims=(256, 256),
            k_iw=20, device=device,
            lambda_adv=0.0, n_critic=0)  # No adversarial (5x faster, same accuracy)
    elif gen_type == 'miwae':
        from generator import MIWAEGenerator
        return MIWAEGenerator(
            input_dim=input_dim, n_classes=n_classes,
            latent_dim=64, hidden_dims=(256, 256),
            k_iw=20, device=device)
    elif gen_type == 'vaeac':
        from vaeac_generator import VAEACGenerator
        return VAEACGenerator(
            input_dim=input_dim, n_classes=n_classes,
            latent_dim=64, hidden_dims=(256, 256),
            k_iw=20, device=device)
    else:
        raise ValueError(f"Unknown generator type: {gen_type}")


def compute_metrics(y_true, y_pred):
    """Compute comprehensive classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    return metrics


def apply_mask(X, mask):
    """Apply a boolean observed-value mask to a copy of X."""
    X_miss = np.asarray(X, dtype=float).copy()
    X_miss[~mask] = np.nan
    return X_miss


def evaluate_with_std(model, X_test, y_test, mask_fn, seeds=(42, 123, 456)):
    """
    Run evaluation over multiple random seeds and return mean +/- std.

    mask_fn(X_test, seed) should return a boolean missingness mask where True
    indicates an observed value. Models that accept both X_miss and mask are
    called with both arguments; plain sklearn-style models fall back to X_miss.
    """
    accs = []
    for seed in seeds:
        np.random.seed(seed)
        mask = mask_fn(X_test, seed=seed)
        X_miss = apply_mask(X_test, mask)
        try:
            y_pred = model.predict(X_miss, mask)
        except TypeError:
            y_pred = model.predict(X_miss)
        accs.append(accuracy_score(y_test, y_pred))
    return np.mean(accs), np.std(accs)


def run_single_experiment(dataset_name, clf_name, miss_key, gen_type, run_id,
                          results_dir='results_v2',
                          preloaded_data=None, pretrained_gen=None):
    """Run a single experiment configuration and return results list.

    Args:
        preloaded_data: Optional tuple (X_train_clean, y_train, X_test_full, y_test, info)
        pretrained_gen: Optional pre-trained generator to reuse
    """

    # Load data (or reuse preloaded)
    if preloaded_data is not None:
        X_train_clean, y_train, X_test_full, y_test, info = preloaded_data
    else:
        X_train, y_train, X_test, y_test, info = DATASET_LOADERS[dataset_name]()
        X_train_clean = X_train.copy()
        X_test_full = X_test.copy()

    assert_clean_training_split(X_train_clean, f"{dataset_name} training split")

    # Train frozen classifier on clean training data
    clf = get_classifier(clf_name)
    clf.fit(X_train_clean, y_train)

    # Use pretrained generator or train new one
    if pretrained_gen is not None:
        gen = pretrained_gen
    else:
        n_classes = len(np.unique(y_train))
        gen = get_generator(gen_type, X_train_clean.shape[1], n_classes)
        if hasattr(gen, 'set_feature_weights'):
            gen.set_feature_weights(clf)
        gen.fit(X_train_clean, y_train, num_epochs=GEN_EPOCHS,
                batch_size=GEN_BATCH_SIZE, lr=GEN_LR)
        gen.prior_class = int(np.bincount(y_train).argmax())

    # Apply missingness
    mechanism, rate = MISSINGNESS[miss_key]
    X_test_miss, mask = apply_missingness(X_test_full, mechanism, rate,
                                          random_state=run_id)
    train_missing_rate = float(np.isnan(X_train_clean).mean())
    test_missing_rate = float(1.0 - mask.mean())

    results = []

    # --- Oracle ---
    t0 = time.perf_counter()
    y_oracle = baseline_oracle(clf, X_test_full)
    t_oracle = time.perf_counter() - t0
    m = compute_metrics(y_test, y_oracle)
    results.append({**m, 'method': 'oracle', 'time': t_oracle})

    # --- Impute-then-predict baselines ---
    for bname, bfunc in IMPUTE_BASELINES.items():
        try:
            t0 = time.perf_counter()
            y_pred = bfunc(clf, X_train_clean, X_test_miss, mask)
            t1 = time.perf_counter() - t0
            m = compute_metrics(y_test, y_pred)
            results.append({**m, 'method': bname, 'time': t1})
        except Exception as e:
            print(f"  WARN: {bname} failed: {e}")

    # --- Native-missing baselines ---
    for bname, bfunc in NATIVE_BASELINES.items():
        try:
            t0 = time.perf_counter()
            y_pred = bfunc(X_train_clean, y_train, X_test_miss)
            t1 = time.perf_counter() - t0
            m = compute_metrics(y_test, y_pred)
            results.append({**m, 'method': bname, 'time': t1})
        except ImportError as e:
            print(f"  SKIP: {bname} (not installed): {e}")
        except Exception as e:
            print(f"  WARN: {bname} failed: {e}")

    # --- Partial distance KNN ---
    try:
        t0 = time.perf_counter()
        y_pred = OTHER_BASELINES['partial_knn'](X_train_clean, y_train,
                                                 X_test_miss, mask)
        t1 = time.perf_counter() - t0
        m = compute_metrics(y_test, y_pred)
        results.append({**m, 'method': 'partial_knn', 'time': t1})
    except Exception as e:
        print(f"  WARN: partial_knn failed: {e}")

    # --- SurrogateML (soft rerouting) ---
    try:
        t0 = time.perf_counter()
        y_surr = wrapped_predict_v2(clf, X_test_miss, mask, gen,
                                    n_mc=N_MC, random_state=run_id)
        t_surr = time.perf_counter() - t0
        m = compute_metrics(y_test, y_surr)
        results.append({**m, 'method': f'surrogate_{gen_type}_soft', 'time': t_surr})
    except Exception as e:
        print(f"  WARN: surrogate soft failed: {e}")
        traceback.print_exc()

    # --- SurrogateML (adaptive MC sampling) ---
    try:
        from steering_v2 import adaptive_soft_forest_predict, adaptive_soft_tree_predict
        from sklearn.tree import DecisionTreeClassifier as _DTC
        from sklearn.ensemble import RandomForestClassifier as _RFC
        if isinstance(clf, (_DTC, _RFC)):
            t0 = time.perf_counter()
            if isinstance(clf, _RFC):
                y_adapt = adaptive_soft_forest_predict(
                    clf, X_test_miss, mask, gen,
                    n_mc_init=20, n_mc_max=200,
                    margin_threshold=0.15, random_state=run_id)
            else:
                # Adaptive for single tree
                preds = []
                for ii, (xo, mm) in enumerate(zip(X_test_miss, mask)):
                    p = adaptive_soft_tree_predict(
                        clf, xo, mm, gen,
                        n_mc_init=20, n_mc_max=200,
                        margin_threshold=0.15, random_state=run_id + ii)
                    preds.append(np.argmax(p))
                y_adapt = np.array(preds)
            t_adapt = time.perf_counter() - t0
            m = compute_metrics(y_test, y_adapt)
            results.append({**m, 'method': f'surrogate_{gen_type}_adaptive', 'time': t_adapt})
    except Exception as e:
        print(f"  WARN: surrogate adaptive failed: {e}")
        traceback.print_exc()

    # --- SurrogateML (legacy hard rerouting, for ablation) ---
    try:
        from steering_legacy import wrapped_predict as wrapped_predict_legacy
        t0 = time.perf_counter()
        y_legacy = wrapped_predict_legacy(clf, X_test_miss, mask, gen,
                                          n_mc=N_MC, random_state=run_id)
        t_legacy = time.perf_counter() - t0
        m = compute_metrics(y_test, y_legacy)
        results.append({**m, 'method': f'surrogate_{gen_type}_hard', 'time': t_legacy})
    except Exception as e:
        print(f"  WARN: surrogate hard failed: {e}")

    # Add metadata to all results
    for r in results:
        r.update({
            'dataset': dataset_name,
            'classifier': clf_name,
            'missingness': miss_key,
            'mechanism': mechanism,
            'rate': rate,
            'generator': gen_type,
            'run': run_id,
            'n_test': len(y_test),
            'n_train': len(y_train),
            'n_features': X_train_clean.shape[1],
            'training_protocol': TRAINING_PROTOCOL,
            'train_missing_rate': train_missing_rate,
            'test_missing_rate_actual': test_missing_rate,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description='SurrogateML Experiment Runner')
    parser.add_argument('--dataset', type=str, default='heart',
                        help='Dataset name or "all"')
    parser.add_argument('--classifier', type=str, default='DT',
                        help='Classifier name or "all"')
    parser.add_argument('--missingness', type=str, default='mcar_30',
                        help='Missingness config key or "all"')
    parser.add_argument('--generator', type=str, default='atvae',
                        help='Generator type: atvae, miwae, vaeac')
    parser.add_argument('--n_runs', type=int, default=10,
                        help='Number of random runs')
    parser.add_argument('--results_dir', type=str, default='results_v2',
                        help='Directory to save results')
    parser.add_argument('--gen_epochs', type=int, default=None,
                        help='Override generator training epochs (default: use GEN_EPOCHS from config)')
    parser.add_argument('--n_mc', type=int, default=None,
                        help='Override MC sample count (default: use N_MC from config)')
    args = parser.parse_args()

    # Override global config if specified
    global GEN_EPOCHS, N_MC
    if args.gen_epochs is not None:
        GEN_EPOCHS = args.gen_epochs
    if args.n_mc is not None:
        N_MC = args.n_mc

    os.makedirs(args.results_dir, exist_ok=True)

    # Resolve "all" or comma-separated lists
    def resolve_arg(val, registry):
        if val == 'all':
            return list(registry.keys())
        return [v.strip() for v in val.split(',')]

    datasets = resolve_arg(args.dataset, DATASET_LOADERS)
    classifiers = resolve_arg(args.classifier, CLASSIFIERS)
    miss_keys = resolve_arg(args.missingness, MISSINGNESS)

    all_results = []
    total_configs = len(datasets) * len(classifiers) * len(miss_keys) * args.n_runs
    completed = 0

    for ds in datasets:
        if ds not in DATASET_LOADERS:
            print(f"SKIP: dataset '{ds}' not found in registry")
            continue

        # Load dataset ONCE
        try:
            X_train, y_train, X_test, y_test, info = DATASET_LOADERS[ds]()
            X_train_clean = X_train.copy()
            X_test_full = X_test.copy()
            preloaded_data = (X_train_clean, y_train, X_test_full, y_test, info)
            print(f"Loaded {ds}: train={X_train_clean.shape}, test={X_test_full.shape}")
        except Exception as e:
            print(f"SKIP: dataset '{ds}' failed to load: {e}")
            completed += len(classifiers) * len(miss_keys) * args.n_runs
            continue

        for run_id in range(args.n_runs):
            # Train generator ONCE per (dataset, run) — reuse across classifiers and missingness
            print(f"Training generator for {ds}/run{run_id} ({GEN_EPOCHS} epochs)...")
            t_gen_start = time.perf_counter()
            n_classes = len(np.unique(y_train))
            gen = get_generator(args.generator, X_train_clean.shape[1], n_classes)
            # Use first classifier for feature weights (DT if available)
            temp_clf = get_classifier(classifiers[0])
            temp_clf.fit(X_train_clean, y_train)
            if hasattr(gen, 'set_feature_weights'):
                gen.set_feature_weights(temp_clf)
            gen.fit(X_train_clean, y_train, num_epochs=GEN_EPOCHS,
                    batch_size=GEN_BATCH_SIZE, lr=GEN_LR)
            gen.prior_class = int(np.bincount(y_train).argmax())
            t_gen = time.perf_counter() - t_gen_start
            print(f"  Generator trained in {t_gen:.1f}s")

            for clf_name in classifiers:
                for miss_key in miss_keys:
                    completed += 1
                    print(f"[{completed}/{total_configs}] "
                          f"{ds}/{clf_name}/{miss_key}/run{run_id} "
                          f"(gen={args.generator})")

                    try:
                        results = run_single_experiment(
                            ds, clf_name, miss_key, args.generator,
                            run_id, args.results_dir,
                            preloaded_data=preloaded_data,
                            pretrained_gen=gen
                        )
                        all_results.extend(results)
                    except FileNotFoundError as e:
                        print(f"  SKIP: {e}")
                        break
                    except Exception as e:
                        print(f"  ERROR: {e}")
                        traceback.print_exc()
                        continue

                # Save incremental results after each classifier's missingness configs
                if all_results:
                    df = pd.DataFrame(all_results)
                    ds_suffix = f"_{ds}" if len(datasets) == 1 else ""
                    csv_path = os.path.join(args.results_dir,
                                            f"results_{args.generator}{ds_suffix}.csv")
                    df.to_csv(csv_path, index=False)

    # Final save
    if all_results:
        df = pd.DataFrame(all_results)
        ds_suffix = f"_{datasets[0]}" if len(datasets) == 1 else ""
        csv_path = os.path.join(args.results_dir, f"results_{args.generator}{ds_suffix}.csv")
        print(f"\nSaved {len(all_results)} result rows to {csv_path}")
        print(f"Unique methods: {df['method'].unique().tolist()}")
        print(f"\nSummary (mean accuracy by method):")
        print(df.groupby('method')['accuracy'].mean().sort_values(ascending=False))
    else:
        print("No results generated.")


if __name__ == '__main__':
    main()
