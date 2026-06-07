"""
Ablation study runners for SurrogateML.

1. ATVAE component ablation (IWAE, cls, adv)
2. MC sample count sweep (T = 1..500)
3. Rerouting strategy comparison (soft vs hard vs impute-mean)
4. Lambda sensitivity (lambda_cls x lambda_adv grid)
5. Generator plug-in comparison (ATVAE vs MIWAE vs VAEAC)

Usage:
    python run_ablations.py --ablation components --dataset heart --n_runs 10
    python run_ablations.py --ablation n_mc --dataset heart --n_runs 10
    python run_ablations.py --ablation rerouting --dataset heart --n_runs 10
    python run_ablations.py --ablation lambda --dataset heart --n_runs 5
    python run_ablations.py --ablation generators --dataset heart --n_runs 10
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.impute import SimpleImputer

from data_loader import DATASETS as DATASET_LOADERS
from missingness import apply_missingness
from steering_v2 import wrapped_predict_v2, soft_tree_predict_batch, soft_forest_predict
from baselines import baseline_mean_impute
from experiment_config import *


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


def compute_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }


def ablation_atvae_components(dataset='heart', clf_name='DT', n_runs=10,
                               results_dir='results_v2'):
    """
    Train 5 generator variants:
    - Full ATVAE (IWAE + cls + adv)
    - No adversarial (IWAE + cls)
    - No task-aware (IWAE + adv)
    - MIWAE only (IWAE only)
    - Standard VAE (ELBO, K=1)
    """
    from novel_generator import AdversarialTaskVAEGenerator

    configs = [
        {'name': 'full_atvae', 'k_iw': 20, 'lambda_cls': 1.0, 'lambda_adv': 1e-3},
        {'name': 'no_adv', 'k_iw': 20, 'lambda_cls': 1.0, 'lambda_adv': 0.0},
        {'name': 'no_cls', 'k_iw': 20, 'lambda_cls': 0.0, 'lambda_adv': 1e-3},
        {'name': 'miwae_only', 'k_iw': 20, 'lambda_cls': 0.0, 'lambda_adv': 0.0},
        {'name': 'standard_vae', 'k_iw': 1, 'lambda_cls': 0.0, 'lambda_adv': 0.0},
    ]

    X_train, y_train, X_test, y_test, info = DATASET_LOADERS[dataset]()
    n_classes = len(np.unique(y_train))
    d = X_train.shape[1]

    clf = get_classifier(clf_name)
    clf.fit(X_train, y_train)

    all_results = []

    for cfg in configs:
        for run_id in range(n_runs):
            print(f"  Component ablation: {cfg['name']} run={run_id}")

            gen = AdversarialTaskVAEGenerator(
                input_dim=d, n_classes=n_classes, latent_dim=64,
                hidden_dims=(256, 256), k_iw=cfg['k_iw'],
                lambda_cls=cfg['lambda_cls'], lambda_adv=cfg['lambda_adv']
            )
            if hasattr(gen, 'set_feature_weights'):
                gen.set_feature_weights(clf)
            gen.fit(X_train, y_train, num_epochs=GEN_EPOCHS,
                    batch_size=GEN_BATCH_SIZE, lr=GEN_LR)
            gen.prior_class = int(np.bincount(y_train).argmax())

            # Test at 30% MCAR
            X_miss, mask = apply_missingness(X_test, 'mcar', 0.3, random_state=run_id)

            t0 = time.perf_counter()
            y_pred = wrapped_predict_v2(clf, X_miss, mask, gen, n_mc=N_MC,
                                        random_state=run_id)
            elapsed = time.perf_counter() - t0
            m = compute_metrics(y_test, y_pred)

            all_results.append({
                **m, 'variant': cfg['name'], 'run': run_id,
                'time': elapsed, 'dataset': dataset, 'classifier': clf_name,
                'ablation': 'components'
            })

    df = pd.DataFrame(all_results)
    path = os.path.join(results_dir, f'ablation_components_{dataset}.csv')
    df.to_csv(path, index=False)
    print(f"Saved to {path}")
    print(df.groupby('variant')['accuracy'].agg(['mean', 'std']))
    return df


def ablation_n_mc(dataset='heart', clf_name='DT', n_runs=10,
                  results_dir='results_v2'):
    """Sweep T (MC samples) in [1, 5, 10, 25, 50, 100, 200, 500]."""
    from novel_generator import AdversarialTaskVAEGenerator

    T_values = [1, 5, 10, 25, 50, 100, 200, 500]

    X_train, y_train, X_test, y_test, info = DATASET_LOADERS[dataset]()
    n_classes = len(np.unique(y_train))
    d = X_train.shape[1]

    clf = get_classifier(clf_name)
    clf.fit(X_train, y_train)

    # Train generator once
    gen = AdversarialTaskVAEGenerator(
        input_dim=d, n_classes=n_classes, latent_dim=64,
        hidden_dims=(256, 256), k_iw=20
    )
    gen.fit(X_train, y_train, num_epochs=GEN_EPOCHS,
            batch_size=GEN_BATCH_SIZE, lr=GEN_LR)
    gen.prior_class = int(np.bincount(y_train).argmax())

    all_results = []

    for T in T_values:
        for run_id in range(n_runs):
            X_miss, mask = apply_missingness(X_test, 'mcar', 0.3, random_state=run_id)

            t0 = time.perf_counter()
            y_pred = wrapped_predict_v2(clf, X_miss, mask, gen, n_mc=T,
                                        random_state=run_id)
            elapsed = time.perf_counter() - t0
            m = compute_metrics(y_test, y_pred)

            all_results.append({
                **m, 'T': T, 'run': run_id, 'time': elapsed,
                'dataset': dataset, 'classifier': clf_name,
                'ablation': 'n_mc'
            })

        print(f"  T={T}: mean_acc={np.mean([r['accuracy'] for r in all_results if r['T']==T]):.4f}")

    df = pd.DataFrame(all_results)
    path = os.path.join(results_dir, f'ablation_nmc_{dataset}.csv')
    df.to_csv(path, index=False)
    print(f"Saved to {path}")
    return df


def ablation_rerouting(dataset='heart', clf_name='DT', n_runs=10,
                       results_dir='results_v2'):
    """Compare rerouting strategies: soft_leaf, hard_node, impute_mean."""
    from novel_generator import AdversarialTaskVAEGenerator

    X_train, y_train, X_test, y_test, info = DATASET_LOADERS[dataset]()
    n_classes = len(np.unique(y_train))
    d = X_train.shape[1]

    clf = get_classifier(clf_name)
    clf.fit(X_train, y_train)

    gen = AdversarialTaskVAEGenerator(
        input_dim=d, n_classes=n_classes, latent_dim=64,
        hidden_dims=(256, 256), k_iw=20
    )
    gen.fit(X_train, y_train, num_epochs=GEN_EPOCHS,
            batch_size=GEN_BATCH_SIZE, lr=GEN_LR)
    gen.prior_class = int(np.bincount(y_train).argmax())

    all_results = []

    for run_id in range(n_runs):
        X_miss, mask = apply_missingness(X_test, 'mcar', 0.3, random_state=run_id)

        # Soft routing (new)
        t0 = time.perf_counter()
        y_soft = wrapped_predict_v2(clf, X_miss, mask, gen, n_mc=N_MC,
                                    random_state=run_id)
        t_soft = time.perf_counter() - t0
        m = compute_metrics(y_test, y_soft)
        all_results.append({**m, 'strategy': 'soft_leaf', 'run': run_id,
                            'time': t_soft, 'dataset': dataset,
                            'classifier': clf_name, 'ablation': 'rerouting'})

        # Hard routing (legacy)
        try:
            from steering_legacy import wrapped_predict as wrapped_predict_legacy
            t0 = time.perf_counter()
            y_hard = wrapped_predict_legacy(clf, X_miss, mask, gen, n_mc=N_MC,
                                            random_state=run_id)
            t_hard = time.perf_counter() - t0
            m = compute_metrics(y_test, y_hard)
            all_results.append({**m, 'strategy': 'hard_node', 'run': run_id,
                                'time': t_hard, 'dataset': dataset,
                                'classifier': clf_name, 'ablation': 'rerouting'})
        except Exception as e:
            print(f"  WARN: hard routing failed: {e}")

        # Mean impute (baseline)
        t0 = time.perf_counter()
        y_mean = baseline_mean_impute(clf, X_train, X_miss, mask)
        t_mean = time.perf_counter() - t0
        m = compute_metrics(y_test, y_mean)
        all_results.append({**m, 'strategy': 'impute_mean', 'run': run_id,
                            'time': t_mean, 'dataset': dataset,
                            'classifier': clf_name, 'ablation': 'rerouting'})

    df = pd.DataFrame(all_results)
    path = os.path.join(results_dir, f'ablation_rerouting_{dataset}.csv')
    df.to_csv(path, index=False)
    print(f"Saved to {path}")
    print(df.groupby('strategy')['accuracy'].agg(['mean', 'std']))
    return df


def ablation_lambda(dataset='heart', clf_name='DT', n_runs=5,
                    results_dir='results_v2'):
    """Grid search lambda_cls x lambda_adv."""
    from novel_generator import AdversarialTaskVAEGenerator

    cls_values = [0.01, 0.1, 1.0, 10.0]
    adv_values = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]

    X_train, y_train, X_test, y_test, info = DATASET_LOADERS[dataset]()
    n_classes = len(np.unique(y_train))
    d = X_train.shape[1]

    clf = get_classifier(clf_name)
    clf.fit(X_train, y_train)

    all_results = []

    for lc in cls_values:
        for la in adv_values:
            for run_id in range(n_runs):
                print(f"  Lambda: cls={lc}, adv={la}, run={run_id}")

                gen = AdversarialTaskVAEGenerator(
                    input_dim=d, n_classes=n_classes, latent_dim=64,
                    hidden_dims=(256, 256), k_iw=20,
                    lambda_cls=lc, lambda_adv=la
                )
                gen.fit(X_train, y_train, num_epochs=GEN_EPOCHS,
                        batch_size=GEN_BATCH_SIZE, lr=GEN_LR)
                gen.prior_class = int(np.bincount(y_train).argmax())

                X_miss, mask = apply_missingness(X_test, 'mcar', 0.3,
                                                  random_state=run_id)
                y_pred = wrapped_predict_v2(clf, X_miss, mask, gen, n_mc=N_MC,
                                            random_state=run_id)
                m = compute_metrics(y_test, y_pred)

                all_results.append({
                    **m, 'lambda_cls': lc, 'lambda_adv': la,
                    'run': run_id, 'dataset': dataset,
                    'classifier': clf_name, 'ablation': 'lambda'
                })

    df = pd.DataFrame(all_results)
    path = os.path.join(results_dir, f'ablation_lambda_{dataset}.csv')
    df.to_csv(path, index=False)
    print(f"Saved to {path}")
    pivot = df.groupby(['lambda_cls', 'lambda_adv'])['accuracy'].mean().unstack()
    print(pivot)
    return df


def ablation_generators(dataset='heart', clf_name='DT', n_runs=10,
                        results_dir='results_v2'):
    """Compare ATVAE vs MIWAE vs VAEAC."""
    from run_experiment import get_generator

    X_train, y_train, X_test, y_test, info = DATASET_LOADERS[dataset]()
    n_classes = len(np.unique(y_train))
    d = X_train.shape[1]

    clf = get_classifier(clf_name)
    clf.fit(X_train, y_train)

    all_results = []

    for gen_type in GENERATORS:
        for run_id in range(n_runs):
            print(f"  Generator: {gen_type}, run={run_id}")

            gen = get_generator(gen_type, d, n_classes)
            gen.fit(X_train, y_train, num_epochs=GEN_EPOCHS,
                    batch_size=GEN_BATCH_SIZE, lr=GEN_LR)
            gen.prior_class = int(np.bincount(y_train).argmax())

            X_miss, mask = apply_missingness(X_test, 'mcar', 0.3,
                                              random_state=run_id)

            t0 = time.perf_counter()
            y_pred = wrapped_predict_v2(clf, X_miss, mask, gen, n_mc=N_MC,
                                        random_state=run_id)
            elapsed = time.perf_counter() - t0
            m = compute_metrics(y_test, y_pred)

            all_results.append({
                **m, 'generator': gen_type, 'run': run_id,
                'time': elapsed, 'dataset': dataset,
                'classifier': clf_name, 'ablation': 'generators'
            })

    df = pd.DataFrame(all_results)
    path = os.path.join(results_dir, f'ablation_generators_{dataset}.csv')
    df.to_csv(path, index=False)
    print(f"Saved to {path}")
    print(df.groupby('generator')['accuracy'].agg(['mean', 'std']))
    return df


ABLATION_MAP = {
    'components': ablation_atvae_components,
    'n_mc': ablation_n_mc,
    'rerouting': ablation_rerouting,
    'lambda': ablation_lambda,
    'generators': ablation_generators,
}


def main():
    parser = argparse.ArgumentParser(description='SurrogateML Ablation Studies')
    parser.add_argument('--ablation', type=str, required=True,
                        choices=list(ABLATION_MAP.keys()),
                        help='Which ablation to run')
    parser.add_argument('--dataset', type=str, default='heart')
    parser.add_argument('--classifier', type=str, default='DT')
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--results_dir', type=str, default='results_v2')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    func = ABLATION_MAP[args.ablation]
    func(dataset=args.dataset, clf_name=args.classifier,
         n_runs=args.n_runs, results_dir=args.results_dir)


if __name__ == '__main__':
    main()
