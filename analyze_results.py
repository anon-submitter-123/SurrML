"""
Results aggregation, statistical tests, and plotting for SurrogateML experiments.

Usage:
    python analyze_results.py --results_dir results_v2 --output_dir figures
    python analyze_results.py --validate --results_dir results_v2
    python analyze_results.py --generate-all --results_dir results_v2 --output_dir figures
"""
import argparse
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import warnings


def load_all_results(results_dir='results_v2'):
    """Load and concatenate all result CSVs."""
    csv_files = glob.glob(os.path.join(results_dir, 'results_*.csv'))
    if not csv_files:
        print(f"No result files found in {results_dir}")
        return pd.DataFrame()

    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(csv_files)} files")
    return df


def load_ablation_results(results_dir='results_v2', ablation_type=None):
    """Load ablation result CSVs."""
    pattern = f'ablation_{ablation_type}_*.csv' if ablation_type else 'ablation_*.csv'
    csv_files = glob.glob(os.path.join(results_dir, pattern))
    if not csv_files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in csv_files]
    return pd.concat(dfs, ignore_index=True)


def compute_summary(df, group_cols=None):
    """Group by columns -> mean, std, count for each metric."""
    if group_cols is None:
        group_cols = ['dataset', 'classifier', 'method', 'missingness']

    metrics = ['accuracy', 'balanced_accuracy', 'macro_f1']
    available_metrics = [m for m in metrics if m in df.columns]

    agg_dict = {m: ['mean', 'std', 'count'] for m in available_metrics}
    if 'time' in df.columns:
        agg_dict['time'] = ['mean', 'std']

    valid_group_cols = [c for c in group_cols if c in df.columns]
    return df.groupby(valid_group_cols).agg(agg_dict).round(4)


def pairwise_wilcoxon(df, method_a, method_b, metric='accuracy'):
    """Wilcoxon signed-rank test comparing two methods across datasets/runs."""
    scores_a = df[df['method'] == method_a].groupby(['dataset', 'run'])[metric].mean()
    scores_b = df[df['method'] == method_b].groupby(['dataset', 'run'])[metric].mean()

    # Align on common indices
    common = scores_a.index.intersection(scores_b.index)
    if len(common) < 5:
        return None, None, len(common)

    a = scores_a.loc[common].values
    b = scores_b.loc[common].values

    try:
        stat, pval = wilcoxon(a, b)
        return stat, pval, len(common)
    except Exception:
        return None, None, len(common)


def validate_results(results_dir='results_v2'):
    """Check results for completeness and issues."""
    df = load_all_results(results_dir)
    if df.empty:
        print("No results to validate.")
        return

    print(f"\n=== Results Validation ===")
    print(f"Total rows: {len(df)}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print(f"Classifiers: {sorted(df['classifier'].unique())}")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Missingness: {sorted(df['missingness'].unique())}")

    # Check for NaN in metrics
    for col in ['accuracy', 'balanced_accuracy', 'macro_f1']:
        if col in df.columns:
            n_nan = df[col].isna().sum()
            if n_nan > 0:
                print(f"WARNING: {n_nan} NaN values in {col}")
            else:
                print(f"OK: No NaN in {col}")

    # Check completeness
    print(f"\nRuns per config:")
    runs = df.groupby(['dataset', 'classifier', 'method', 'missingness']).size()
    print(f"  Min: {runs.min()}, Max: {runs.max()}, Median: {runs.median()}")

    # Summary table
    summary = compute_summary(df)
    print(f"\n=== Summary Table ===")
    print(summary.to_string())


def plot_missingness_sweep(df, dataset, classifier, metric='accuracy',
                           output_dir='figures'):
    """Plot accuracy vs missingness rate, one curve per method."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    subset = df[(df['dataset'] == dataset) & (df['classifier'] == classifier)]
    if subset.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, mechanism in zip(axes, ['mcar', 'mar', 'mnar']):
        mech_data = subset[subset['mechanism'] == mechanism]
        if mech_data.empty:
            continue

        methods = mech_data['method'].unique()
        for method in sorted(methods):
            method_data = mech_data[mech_data['method'] == method]
            grouped = method_data.groupby('rate')[metric].agg(['mean', 'std'])
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                        label=method, marker='o', capsize=3)

        ax.set_title(f'{mechanism.upper()}')
        ax.set_xlabel('Missingness Rate')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.legend(fontsize=7, loc='lower left')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{dataset} / {classifier}', fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, f'sweep_{dataset}_{classifier}.pdf')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_ablation_bars(df, ablation_type, output_dir='figures'):
    """Bar chart for ablation study results."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        return

    os.makedirs(output_dir, exist_ok=True)

    if ablation_type == 'components':
        group_col = 'variant'
    elif ablation_type == 'rerouting':
        group_col = 'strategy'
    elif ablation_type == 'generators':
        group_col = 'generator'
    elif ablation_type == 'n_mc':
        group_col = 'T'
    else:
        return

    grouped = df.groupby(group_col)['accuracy'].agg(['mean', 'std'])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(grouped))
    ax.bar(x, grouped['mean'], yerr=grouped['std'], capsize=4, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index, rotation=30, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Ablation: {ablation_type}')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(output_dir, f'ablation_{ablation_type}.pdf')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_lambda_heatmap(df, output_dir='figures'):
    """Lambda sensitivity heatmap."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        return

    os.makedirs(output_dir, exist_ok=True)

    pivot = df.groupby(['lambda_cls', 'lambda_adv'])['accuracy'].mean().unstack()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
    ax.set_xlabel('lambda_adv')
    ax.set_ylabel('lambda_cls')
    ax.set_title('Lambda Sensitivity (Accuracy)')

    plt.tight_layout()
    path = os.path.join(output_dir, 'ablation_lambda_heatmap.pdf')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def generate_all(results_dir='results_v2', output_dir='figures'):
    """Generate all figures."""
    os.makedirs(output_dir, exist_ok=True)

    # Main results
    df = load_all_results(results_dir)
    if not df.empty:
        for dataset in df['dataset'].unique():
            for clf in df['classifier'].unique():
                plot_missingness_sweep(df, dataset, clf, output_dir=output_dir)

    # Ablation results
    for abl_type in ['components', 'rerouting', 'generators', 'n_mc']:
        abl_df = load_ablation_results(results_dir, abl_type)
        if not abl_df.empty:
            plot_ablation_bars(abl_df, abl_type, output_dir=output_dir)

    # Lambda heatmap
    lambda_df = load_ablation_results(results_dir, 'lambda')
    if not lambda_df.empty:
        plot_lambda_heatmap(lambda_df, output_dir=output_dir)

    # Statistical tests
    if not df.empty:
        print("\n=== Pairwise Wilcoxon Tests ===")
        methods = df['method'].unique()
        surrogate_methods = [m for m in methods if 'surrogate' in m]
        baseline_methods = [m for m in methods if 'surrogate' not in m and m != 'oracle']

        for sm in surrogate_methods:
            for bm in baseline_methods:
                stat, pval, n = pairwise_wilcoxon(df, sm, bm)
                if pval is not None:
                    sig = '*' if pval < 0.05 else ''
                    print(f"  {sm} vs {bm}: p={pval:.4f}{sig}  (n={n})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results_v2')
    parser.add_argument('--output_dir', type=str, default='figures')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--generate-all', action='store_true')
    args = parser.parse_args()

    if args.validate:
        validate_results(args.results_dir)
    elif args.generate_all:
        generate_all(args.results_dir, args.output_dir)
    else:
        validate_results(args.results_dir)
        generate_all(args.results_dir, args.output_dir)


if __name__ == '__main__':
    main()
