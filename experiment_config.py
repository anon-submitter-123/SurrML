"""Central configuration for all SurrogateML experiments."""

# Classifiers to test SurrogateML with
CLASSIFIERS = {
    'DT': ('DecisionTreeClassifier', dict(max_depth=5, random_state=42)),
    'RF': ('RandomForestClassifier', dict(n_estimators=100, random_state=42, class_weight='balanced')),
    'KNN': ('KNeighborsClassifier', dict(n_neighbors=5)),
    'LogReg': ('LogisticRegression', dict(random_state=42, max_iter=1000)),
    'MLP': ('MLPClassifier', dict(hidden_layer_sizes=(128, 64), random_state=42, max_iter=500)),
}

# Datasets to evaluate
DATASETS = ['heart', 'mushroom', 'breast_cancer', 'adult', 'credit', 'letter', 'california', 'banknote']

# Missingness configs: key -> (mechanism, rate)
MISSINGNESS = {
    'mcar_10': ('mcar', 0.10),
    'mcar_30': ('mcar', 0.30),
    'mcar_50': ('mcar', 0.50),
    'mar_10': ('mar', 0.10),
    'mar_30': ('mar', 0.30),
    'mar_50': ('mar', 0.50),
    'mnar_10': ('mnar', 0.10),
    'mnar_30': ('mnar', 0.30),
    'mnar_50': ('mnar', 0.50),
}

# Generator configs for ablation
GENERATORS = ['atvae', 'miwae', 'vaeac']

# Number of runs per experiment
N_RUNS = 10

# MC sample counts
N_MC = 100

# Generator training params
GEN_EPOCHS = 200
GEN_BATCH_SIZE = 128
GEN_LR = 1e-3
